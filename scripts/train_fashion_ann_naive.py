from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FashionMNIST


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_fashion_arrays(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_ds = FashionMNIST(root=str(root), train=True, download=True)
    test_ds = FashionMNIST(root=str(root), train=False, download=True)
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


class RawPixelDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = images.astype(np.float32).reshape(len(labels), -1) / 255.0
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return image, label


class NaiveFashionAnn(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
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

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
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
) -> tuple[float, np.ndarray, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_correct = 0
    total_samples = 0
    confusion = np.zeros((10, 10), dtype=np.int64)
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()
            probs_np = probabilities.cpu().numpy()
            all_targets.append(labels_np)
            all_predictions.append(preds_np)
            all_probabilities.append(probs_np)

            for truth, pred in zip(labels_np, preds_np, strict=True):
                confusion[truth, pred] += 1

    per_class_accuracy: dict[str, float] = {}
    for class_idx in range(10):
        total_in_class = confusion[class_idx].sum()
        accuracy = float(confusion[class_idx, class_idx] / total_in_class) if total_in_class > 0 else 0.0
        per_class_accuracy[str(class_idx)] = accuracy

    return (
        total_correct / max(total_samples, 1),
        confusion,
        per_class_accuracy,
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_predictions, axis=0),
        np.concatenate(all_probabilities, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a naive ANN baseline on raw FashionMNIST pixels.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fashion_mnist_ann_naive"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    train_images, train_labels, test_images, test_labels = load_fashion_arrays(args.data_root)
    train_images, train_labels = maybe_limit_split(train_images, train_labels, args.limit_train)
    test_images, test_labels = maybe_limit_split(test_images, test_labels, args.limit_test)

    train_ds = RawPixelDataset(train_images, train_labels)
    test_ds = RawPixelDataset(test_images, test_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = NaiveFashionAnn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        row = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_acc": train_result.accuracy,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    final_train_acc = history[-1]["train_acc"]
    final_train_loss = history[-1]["train_loss"]
    final_test_acc, confusion, per_class_accuracy, targets, predictions, probabilities = evaluate_test(model, test_loader, device)
    generalization_gap = final_train_acc - final_test_acc

    torch.save(
        {
            "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "args": vars(args),
            "final_train_acc": final_train_acc,
            "final_train_loss": final_train_loss,
            "test_acc": final_test_acc,
            "generalization_gap": generalization_gap,
            "per_class_accuracy": per_class_accuracy,
        },
        args.output_dir / "model_final.pt",
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
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "class_accuracy.json").write_text(
        json.dumps(per_class_accuracy, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.save(args.output_dir / "confusion_matrix.npy", confusion)
    np.save(args.output_dir / "test_targets.npy", targets)
    np.save(args.output_dir / "test_predictions.npy", predictions)
    np.save(args.output_dir / "test_probabilities.npy", probabilities)

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


if __name__ == "__main__":
    main()
