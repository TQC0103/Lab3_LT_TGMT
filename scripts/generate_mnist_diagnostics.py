from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader

import train_mnist_ann as two_branch_module
import train_mnist_ann_naive as naive_module
import train_mnist_ann_single_branch as single_branch_module


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
FIGURES_ROOT = REPO_ROOT / "report" / "figures" / "mnist-models"


MODEL_CONFIGS = [
    {
        "slug": "naive",
        "title": "Naive ANN",
        "artifact_dir": ARTIFACTS_ROOT / "mnist_ann_naive",
        "checkpoint": "model_final.pt",
        "kind": "naive",
    },
    {
        "slug": "single_branch",
        "title": "Single-branch ANN",
        "artifact_dir": ARTIFACTS_ROOT / "mnist_ann_single_branch",
        "checkpoint": "model_final.pt",
        "kind": "single_branch",
    },
    {
        "slug": "single_branch_augmented",
        "title": "Single-branch + local augmentation",
        "artifact_dir": ARTIFACTS_ROOT / "mnist_ann_single_branch_augmented",
        "checkpoint": "model_final.pt",
        "kind": "single_branch",
    },
    {
        "slug": "two_branch",
        "title": "Two-branch ANN",
        "artifact_dir": ARTIFACTS_ROOT / "mnist_ann_two_branch",
        "checkpoint": "best_model.pt",
        "kind": "two_branch",
    },
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def get_test_images_and_labels(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    _, _, test_images, test_labels = naive_module.load_mnist_arrays(data_root)
    return test_images, test_labels


def build_naive_outputs(checkpoint: dict, test_images: np.ndarray, test_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = naive_module.NaiveAnn(input_dim=28 * 28)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_ds = naive_module.RawPixelDataset(test_images, test_labels)
    loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    probs_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    with torch.no_grad():
        for images, _labels in loader:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())
            preds_list.append(probs.argmax(dim=1).cpu().numpy())

    return np.concatenate(preds_list), np.concatenate(probs_list)


def build_single_branch_outputs(checkpoint: dict, test_images: np.ndarray, test_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    args = single_branch_module.argparse.Namespace(**checkpoint["args"])
    cache = single_branch_module.load_or_create_cache(
        "diagnostic_test",
        test_images,
        test_labels,
        Path(args.cache_dir),
        args=args,
        augment=False,
    )
    mean = checkpoint["feature_mean"]
    std = checkpoint["feature_std"]
    test_ds = single_branch_module.FeatureDataset(cache["features"], cache["labels"], mean, std)
    loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = single_branch_module.SingleBranchAnn(input_dim=cache["features"].shape[1], dropout=args.dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probs_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    with torch.no_grad():
        for features, _labels in loader:
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())
            preds_list.append(probs.argmax(dim=1).cpu().numpy())

    return np.concatenate(preds_list), np.concatenate(probs_list)


def build_two_branch_outputs(checkpoint: dict, test_images: np.ndarray, test_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    args = two_branch_module.argparse.Namespace(**checkpoint["args"])
    cache = two_branch_module.load_or_create_cache(
        "diagnostic_test",
        test_images,
        test_labels,
        Path(args.cache_dir),
        args=args,
        augment=False,
    )
    raw_mean = checkpoint["raw_mean"]
    raw_std = checkpoint["raw_std"]
    hog_mean = checkpoint["hog_mean"]
    hog_std = checkpoint["hog_std"]

    test_ds = two_branch_module.FeatureDataset(
        cache["raw"],
        cache["hog"],
        cache["labels"],
        raw_mean,
        raw_std,
        hog_mean,
        hog_std,
    )
    loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = two_branch_module.MnistAnn(raw_dim=cache["raw"].shape[1], hog_dim=cache["hog"].shape[1], dropout=args.dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probs_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    with torch.no_grad():
        for raw_x, hog_x, _labels in loader:
            logits = model(raw_x, hog_x)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())
            preds_list.append(probs.argmax(dim=1).cpu().numpy())

    return np.concatenate(preds_list), np.concatenate(probs_list)


def save_predictions(artifact_dir: Path, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> None:
    np.save(artifact_dir / "test_targets.npy", labels.astype(np.int64))
    np.save(artifact_dir / "test_predictions.npy", preds.astype(np.int64))
    np.save(artifact_dir / "test_probabilities.npy", probs.astype(np.float32))


def plot_training_curves(title: str, history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(epochs, train_acc, color="#1d3557", linewidth=2)
    axes[0].set_title(f"{title} - Train accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, train_loss, color="#c1121f", linewidth=2)
    axes[1].set_title(f"{title} - Train loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(confusion: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(confusion, cmap="Blues", norm=Normalize(vmin=0, vmax=confusion.max()))
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified_gallery(
    title: str,
    test_images: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
    max_examples: int = 12,
) -> None:
    wrong_indices = np.flatnonzero(preds != labels)
    if len(wrong_indices) == 0:
        return

    confidences = probs[wrong_indices, preds[wrong_indices]]
    ranked = wrong_indices[np.argsort(-confidences)[:max_examples]]

    fig, axes = plt.subplots(3, 4, figsize=(9.6, 7.2))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, ranked, strict=False):
        ax.imshow(test_images[idx], cmap="gray")
        ax.set_title(
            f"t={labels[idx]}, p={preds[idx]}\nconf={probs[idx, preds[idx]]:.3f}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_class_specific_errors(
    title: str,
    test_images: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
    target_class: int,
    max_examples: int = 12,
) -> None:
    wrong_indices = np.flatnonzero((labels == target_class) & (preds != labels))
    if len(wrong_indices) == 0:
        return

    confidences = probs[wrong_indices, preds[wrong_indices]]
    ranked = wrong_indices[np.argsort(-confidences)[:max_examples]]

    fig, axes = plt.subplots(3, 4, figsize=(9.6, 7.2))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, ranked, strict=False):
        ax.imshow(test_images[idx], cmap="gray")
        ax.set_title(
            f"7 -> {preds[idx]}\nconf={probs[idx, preds[idx]]:.3f}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(summary_rows: list[dict[str, float | str]], output_path: Path) -> None:
    labels = [row["title"] for row in summary_rows]
    test_acc = [float(row["test_acc"]) for row in summary_rows]
    gap = [float(row["generalization_gap"]) for row in summary_rows]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(10.2, 4.8))
    bars1 = ax1.bar(x - width / 2, test_acc, width, label="Test accuracy", color="#1d3557")
    bars2 = ax1.bar(x + width / 2, gap, width, label="Generalization gap", color="#c1121f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=10, ha="right")
    ax1.set_ylabel("Value")
    ax1.set_title("Comparison of the four MNIST ANN variants")
    ax1.grid(axis="y", alpha=0.2)
    ax1.legend()

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.001, f"{height:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_accuracy(summary_rows: list[dict[str, float | str]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    classes = np.arange(10)

    for row in summary_rows:
        per_class = row["per_class_accuracy"]
        acc = [per_class[str(idx)] for idx in range(10)]
        ax.plot(classes, acc, marker="o", linewidth=1.8, label=row["title"])

    ax.set_xticks(classes)
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.9, 1.001)
    ax.set_title("Per-class accuracy across MNIST ANN variants")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir(FIGURES_ROOT)
    data_root = REPO_ROOT / "data"
    test_images, test_labels = get_test_images_and_labels(data_root)

    summary_rows: list[dict[str, float | str | dict[str, float]]] = []

    for config in MODEL_CONFIGS:
        artifact_dir = config["artifact_dir"]
        metrics = load_json(artifact_dir / "metrics.json")
        checkpoint = load_checkpoint(artifact_dir / config["checkpoint"])
        confusion = np.load(artifact_dir / "confusion_matrix.npy")

        if config["kind"] == "naive":
            preds, probs = build_naive_outputs(checkpoint, test_images, test_labels)
        elif config["kind"] == "single_branch":
            preds, probs = build_single_branch_outputs(checkpoint, test_images, test_labels)
        else:
            preds, probs = build_two_branch_outputs(checkpoint, test_images, test_labels)

        save_predictions(artifact_dir, test_labels, preds, probs)

        plot_training_curves(
            config["title"],
            metrics["history"],
            FIGURES_ROOT / f"{config['slug']}_training_curves.png",
        )
        plot_confusion(
            confusion,
            f"{config['title']} confusion matrix",
            FIGURES_ROOT / f"{config['slug']}_confusion_matrix.png",
        )
        plot_misclassified_gallery(
            f"{config['title']} - high-confidence misclassifications",
            test_images,
            test_labels,
            preds,
            probs,
            FIGURES_ROOT / f"{config['slug']}_misclassified_gallery.png",
        )
        if config["slug"] == "naive":
            plot_class_specific_errors(
                "Naive ANN - misclassified samples from class 7",
                test_images,
                test_labels,
                preds,
                probs,
                FIGURES_ROOT / "naive_class7_error_gallery.png",
                target_class=7,
            )

        summary_rows.append(
            {
                "title": config["title"],
                "test_acc": metrics["test_acc"],
                "generalization_gap": metrics.get("generalization_gap", metrics["final_train_acc"] - metrics["test_acc"]),
                "per_class_accuracy": metrics["per_class_accuracy"],
            }
        )

    plot_model_comparison(summary_rows, FIGURES_ROOT / "model_comparison.png")
    plot_per_class_accuracy(summary_rows, FIGURES_ROOT / "per_class_accuracy_comparison.png")


if __name__ == "__main__":
    main()
