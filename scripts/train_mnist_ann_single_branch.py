from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.feature import hog
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FashionMNIST, MNIST


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_slant_angle(image: np.ndarray, threshold: float = 0.1) -> float:
    mask = image > threshold
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return 0.0

    weights = image[ys, xs].astype(np.float64)
    if weights.sum() <= 1e-8:
        return 0.0

    x_mean = np.average(xs, weights=weights)
    y_mean = np.average(ys, weights=weights)
    x_centered = xs - x_mean
    y_centered = ys - y_mean

    var_y = np.average(y_centered * y_centered, weights=weights)
    if var_y <= 1e-8:
        return 0.0

    cov_xy = np.average(x_centered * y_centered, weights=weights)
    return math.atan(cov_xy / var_y)


def deslant_image(image: np.ndarray) -> np.ndarray:
    angle = compute_slant_angle(image)
    shear = -math.tan(angle)
    if abs(shear) < 1e-4:
        return image.copy()

    h, w = image.shape
    center_y = (h - 1) / 2.0
    tx = -shear * center_y
    matrix = np.array([[1.0, shear, tx], [0.0, 1.0, 0.0]], dtype=np.float32)
    warped = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.clip(warped, 0.0, 1.0)


def recenter_image(image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    mask = image > threshold
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return image.copy()

    weights = image[ys, xs].astype(np.float64)
    if weights.sum() <= 1e-8:
        return image.copy()

    x_mean = np.average(xs, weights=weights)
    y_mean = np.average(ys, weights=weights)
    target_x = (image.shape[1] - 1) / 2.0
    target_y = (image.shape[0] - 1) / 2.0
    shift_x = target_x - x_mean
    shift_y = target_y - y_mean

    matrix = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    warped = cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.clip(warped, 0.0, 1.0)


def random_affine(image: np.ndarray, max_rotate: float, max_shift: float, max_shear: float) -> np.ndarray:
    angle = random.uniform(-max_rotate, max_rotate)
    shear_deg = random.uniform(-max_shear, max_shear)
    tx = random.uniform(-max_shift, max_shift) * image.shape[1]
    ty = random.uniform(-max_shift, max_shift) * image.shape[0]

    center = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot[0, 2] += tx
    rot[1, 2] += ty
    warped = cv2.warpAffine(
        image,
        rot,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    shear = math.tan(math.radians(shear_deg))
    center_y = (image.shape[0] - 1) / 2.0
    tx_shear = -shear * center_y
    shear_matrix = np.array([[1.0, shear, tx_shear], [0.0, 1.0, 0.0]], dtype=np.float32)
    warped = cv2.warpAffine(
        warped,
        shear_matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.clip(warped, 0.0, 1.0)


def elastic_distort(image: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    if alpha <= 0.0 or sigma <= 0.0:
        return image.copy()

    h, w = image.shape
    dx = np.random.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
    dy = np.random.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma) * alpha
    dy = cv2.GaussianBlur(dy, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma) * alpha

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(grid_x + dx, 0.0, w - 1.0)
    map_y = np.clip(grid_y + dy, 0.0, h - 1.0)
    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.clip(warped, 0.0, 1.0)


def stroke_jitter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return image.copy()

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    binary = (image * 255.0).astype(np.uint8)
    operation = random.choice((cv2.MORPH_ERODE, cv2.MORPH_DILATE))
    jittered = cv2.morphologyEx(binary, operation, kernel)
    return np.clip(jittered.astype(np.float32) / 255.0, 0.0, 1.0)


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True,
    )
    return features.astype(np.float32)


def preprocess_image(image: np.ndarray, use_augmentation: bool, args: argparse.Namespace) -> np.ndarray:
    proc = image.astype(np.float32) / 255.0
    if use_augmentation:
        proc = random_affine(
            proc,
            max_rotate=args.max_rotate_deg,
            max_shift=args.max_shift_ratio,
            max_shear=args.max_shear_deg,
        )
        if random.random() < args.elastic_prob:
            proc = elastic_distort(proc, alpha=args.elastic_alpha, sigma=args.elastic_sigma)

    proc = deslant_image(proc)
    proc = recenter_image(proc)
    if use_augmentation and random.random() < args.stroke_jitter_prob:
        proc = stroke_jitter(proc, kernel_size=args.stroke_kernel_size)
    raw = proc.reshape(-1).astype(np.float32)
    hog_feat = extract_hog_features(proc)
    return np.concatenate([raw, hog_feat], axis=0).astype(np.float32)


def build_feature_cache(
    images: np.ndarray,
    labels: np.ndarray,
    cache_path: Path,
    args: argparse.Namespace,
    augment: bool,
) -> dict[str, np.ndarray]:
    features = [preprocess_image(image, use_augmentation=augment, args=args) for image in images]
    cache = {
        "features": np.stack(features).astype(np.float32),
        "labels": labels.astype(np.int64),
    }
    np.savez_compressed(cache_path, **cache)
    return cache


def load_or_create_cache(
    split_name: str,
    images: np.ndarray,
    labels: np.ndarray,
    cache_dir: Path,
    args: argparse.Namespace,
    augment: bool,
) -> dict[str, np.ndarray]:
    suffix = "aug" if augment else "plain"
    cache_path = cache_dir / f"{split_name}_{suffix}_{build_cache_tag(args, augment)}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return {key: data[key] for key in data.files}
    return build_feature_cache(images, labels, cache_path, args=args, augment=augment)


def format_cache_value(value: Any) -> str:
    if isinstance(value, float):
        return str(value).replace(".", "p")
    return str(value)


def build_cache_tag(args: argparse.Namespace, augment: bool) -> str:
    parts = [f"seed-{args.seed}"]
    if augment:
        parts.extend(
            [
                f"rot-{format_cache_value(args.max_rotate_deg)}",
                f"shift-{format_cache_value(args.max_shift_ratio)}",
                f"shear-{format_cache_value(args.max_shear_deg)}",
                f"eprob-{format_cache_value(args.elastic_prob)}",
                f"ealpha-{format_cache_value(args.elastic_alpha)}",
                f"esigma-{format_cache_value(args.elastic_sigma)}",
                f"sprob-{format_cache_value(args.stroke_jitter_prob)}",
                f"skernel-{format_cache_value(args.stroke_kernel_size)}",
            ]
        )
    return "_".join(parts)


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        self.features = (features - mean) / std
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature = torch.from_numpy(self.features[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return feature, label


class SingleBranchAnn(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochResult:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, labels)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return EpochResult(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
    )


def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, dict[str, float]]:
    model.eval()
    total_correct = 0
    total_samples = 0
    confusion = np.zeros((10, 10), dtype=np.int64)

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            preds = logits.argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            for truth, pred in zip(labels.cpu().numpy(), preds.cpu().numpy(), strict=True):
                confusion[truth, pred] += 1

    per_class_accuracy: dict[str, float] = {}
    for class_idx in range(10):
        total_in_class = confusion[class_idx].sum()
        accuracy = float(confusion[class_idx, class_idx] / total_in_class) if total_in_class > 0 else 0.0
        per_class_accuracy[str(class_idx)] = accuracy

    return total_correct / max(total_samples, 1), confusion, per_class_accuracy


def resolve_dataset(name: str):
    dataset_name = name.strip().lower()
    if dataset_name == "mnist":
        return MNIST, "mnist"
    if dataset_name in {"fashion", "fashionmnist", "fashion-mnist"}:
        return FashionMNIST, "fashion_mnist"
    raise ValueError(f"Unsupported dataset: {name}")


def load_dataset_arrays(root: Path, dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_cls, _ = resolve_dataset(dataset_name)
    train_ds = dataset_cls(root=str(root), train=True, download=True)
    test_ds = dataset_cls(root=str(root), train=False, download=True)
    return (
        train_ds.data.numpy(),
        train_ds.targets.numpy(),
        test_ds.data.numpy(),
        test_ds.targets.numpy(),
    )


def maybe_limit_split(images: np.ndarray, labels: np.ndarray, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or limit <= 0 or limit >= len(labels):
        return images, labels
    return images[:limit], labels[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single-branch ANN on preprocessed MNIST or FashionMNIST features.")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rotate-deg", type=float, default=10.0)
    parser.add_argument("--max-shift-ratio", type=float, default=0.08)
    parser.add_argument("--max-shear-deg", type=float, default=8.0)
    parser.add_argument("--elastic-prob", type=float, default=0.0)
    parser.add_argument("--elastic-alpha", type=float, default=0.0)
    parser.add_argument("--elastic-sigma", type=float, default=0.0)
    parser.add_argument("--stroke-jitter-prob", type=float, default=0.0)
    parser.add_argument("--stroke-kernel-size", type=int, default=2)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    _, dataset_slug = resolve_dataset(args.dataset)
    if args.cache_dir is None:
        args.cache_dir = Path(f"data/{dataset_slug}_ann_single_branch_cache")
    if args.output_dir is None:
        args.output_dir = Path(f"artifacts/{dataset_slug}_ann_single_branch")

    set_seed(args.seed)
    ensure_dir(args.cache_dir)
    ensure_dir(args.output_dir)

    train_images, train_labels, test_images, test_labels = load_dataset_arrays(args.data_root, args.dataset)
    train_images, train_labels = maybe_limit_split(train_images, train_labels, args.limit_train)
    test_images, test_labels = maybe_limit_split(test_images, test_labels, args.limit_test)

    train_cache = load_or_create_cache("train", train_images, train_labels, args.cache_dir, args=args, augment=True)
    test_cache = load_or_create_cache("test", test_images, test_labels, args.cache_dir, args=args, augment=False)

    mean = train_cache["features"].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_cache["features"].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    train_ds = FeatureDataset(train_cache["features"], train_cache["labels"], mean, std)
    test_ds = FeatureDataset(test_cache["features"], test_cache["labels"], mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = SingleBranchAnn(input_dim=train_cache["features"].shape[1], dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_acc": train_result.accuracy,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    final_train_acc = history[-1]["train_acc"]
    final_train_loss = history[-1]["train_loss"]
    final_test_acc, confusion, per_class_accuracy = evaluate_test(model, test_loader, device)
    generalization_gap = final_train_acc - final_test_acc

    model_path = args.output_dir / "model_final.pt"
    metrics_path = args.output_dir / "metrics.json"
    confusion_path = args.output_dir / "confusion_matrix.npy"
    class_accuracy_path = args.output_dir / "class_accuracy.json"

    torch.save(
        {
            "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "feature_mean": mean,
            "feature_std": std,
            "args": vars(args),
            "final_train_acc": final_train_acc,
            "final_train_loss": final_train_loss,
            "test_acc": final_test_acc,
            "generalization_gap": generalization_gap,
            "per_class_accuracy": per_class_accuracy,
        },
        model_path,
    )

    metrics = {
        "final_train_acc": final_train_acc,
        "final_train_loss": final_train_loss,
        "test_acc": final_test_acc,
        "generalization_gap": generalization_gap,
        "per_class_accuracy": per_class_accuracy,
        "epochs_ran": len(history),
        "history": history,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    class_accuracy_path.write_text(json.dumps(per_class_accuracy, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(confusion_path, confusion)

    print(
        json.dumps(
            {
                "final_train_acc": final_train_acc,
                "test_acc": final_test_acc,
                "generalization_gap": generalization_gap,
            },
            ensure_ascii=False,
        )
    )
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {confusion_path}")
    print(f"Saved per-class accuracy to: {class_accuracy_path}")


if __name__ == "__main__":
    main()
