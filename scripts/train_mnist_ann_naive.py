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
from torchvision.datasets import MNIST


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_mnist_arrays(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_ds = MNIST(root=str(root), train=True, download=True)
    test_ds = MNIST(root=str(root), train=False, download=True)
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


class NaiveAnn(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
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
) -> tuple[float, np.ndarray, dict[str, float]]:
    model.eval()
    total_correct = 0
    total_samples = 0
    confusion = np.zeros((10, 10), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the most naive ANN baseline on raw MNIST pixels.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mnist_ann_naive"))
    parser.add_argument("--epochs", type=int, default=20)
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

    train_images, train_labels, test_images, test_labels = load_mnist_arrays(args.data_root)
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

    model = NaiveAnn(input_dim=28 * 28).to(device)
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
    final_test_acc, confusion, per_class_accuracy = evaluate_test(model, test_loader, device)
    generalization_gap = final_train_acc - final_test_acc

    model_path = args.output_dir / "model_final.pt"
    metrics_path = args.output_dir / "metrics.json"
    confusion_path = args.output_dir / "confusion_matrix.npy"
    class_accuracy_path = args.output_dir / "class_accuracy.json"

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
