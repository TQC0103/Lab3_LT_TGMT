from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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
    w_sum = weights.sum()
    if w_sum <= 1e-8:
        return 0.0

    x_mean = np.average(xs, weights=weights)
    y_mean = np.average(ys, weights=weights)
    x_centered = xs - x_mean
    y_centered = ys - y_mean

    var_y = np.average(y_centered * y_centered, weights=weights)
    if var_y <= 1e-8:
        return 0.0

    cov_xy = np.average(x_centered * y_centered, weights=weights)
    beta = cov_xy / var_y
    return math.atan(beta)


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
    w_sum = weights.sum()
    if w_sum <= 1e-8:
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


def preprocess_image(image: np.ndarray, use_augmentation: bool, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    proc = image.astype(np.float32) / 255.0
    if use_augmentation:
        proc = random_affine(
            proc,
            max_rotate=args.max_rotate_deg,
            max_shift=args.max_shift_ratio,
            max_shear=args.max_shear_deg,
        )

    proc = deslant_image(proc)
    proc = recenter_image(proc)
    raw = proc.reshape(-1).astype(np.float32)
    hog_feat = extract_hog_features(proc)
    return raw, hog_feat


def build_feature_cache(
    images: np.ndarray,
    labels: np.ndarray,
    cache_path: Path,
    args: argparse.Namespace,
    augment: bool,
) -> dict[str, np.ndarray]:
    raw_list = []
    hog_list = []
    for image in images:
        raw, hog_feat = preprocess_image(image, use_augmentation=augment, args=args)
        raw_list.append(raw)
        hog_list.append(hog_feat)

    cache = {
        "raw": np.stack(raw_list).astype(np.float32),
        "hog": np.stack(hog_list).astype(np.float32),
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
    cache_path = cache_dir / f"{split_name}_{suffix}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return {key: data[key] for key in data.files}
    return build_feature_cache(images, labels, cache_path, args=args, augment=augment)


class FeatureDataset(Dataset):
    def __init__(
        self,
        raw_features: np.ndarray,
        hog_features: np.ndarray,
        labels: np.ndarray,
        raw_mean: np.ndarray,
        raw_std: np.ndarray,
        hog_mean: np.ndarray,
        hog_std: np.ndarray,
    ) -> None:
        self.raw = (raw_features - raw_mean) / raw_std
        self.hog = (hog_features - hog_mean) / hog_std
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = torch.from_numpy(self.raw[idx]).float()
        hog_feat = torch.from_numpy(self.hog[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return raw, hog_feat, label


class Branch(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out_dim = current_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MnistAnn(nn.Module):
    def __init__(self, raw_dim: int, hog_dim: int, dropout: float) -> None:
        super().__init__()
        self.raw_branch = Branch(raw_dim, [512, 256], dropout)
        self.hog_branch = Branch(hog_dim, [768, 256], dropout)
        fused_dim = self.raw_branch.out_dim + self.hog_branch.out_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, raw_x: torch.Tensor, hog_x: torch.Tensor) -> torch.Tensor:
        raw_feat = self.raw_branch(raw_x)
        hog_feat = self.hog_branch(hog_x)
        fused = torch.cat([raw_feat, hog_feat], dim=1)
        return self.head(fused)


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

    for raw_x, hog_x, labels in loader:
        raw_x = raw_x.to(device)
        hog_x = hog_x.to(device)
        labels = labels.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        logits = model(raw_x, hog_x)
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
        for raw_x, hog_x, labels in loader:
            raw_x = raw_x.to(device)
            hog_x = hog_x.to(device)
            labels = labels.to(device)
            logits = model(raw_x, hog_x)
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
    normalized = name.strip().lower()
    if normalized == "mnist":
        return MNIST, "mnist"
    if normalized in {"fashionmnist", "fashion-mnist", "fashion_mnist"}:
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
    parser = argparse.ArgumentParser(description="Train a non-CNN two-branch ANN using deslant + HOG features.")
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
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    _, dataset_slug = resolve_dataset(args.dataset)
    if args.cache_dir is None:
        args.cache_dir = Path(f"data/{dataset_slug}_ann_two_branch_cache")
    if args.output_dir is None:
        args.output_dir = Path(f"artifacts/{dataset_slug}_ann_two_branch")

    set_seed(args.seed)
    ensure_dir(args.cache_dir)
    ensure_dir(args.output_dir)

    train_images, train_labels, test_images, test_labels = load_dataset_arrays(args.data_root, args.dataset)
    x_train, y_train = maybe_limit_split(train_images, train_labels, args.limit_train)
    test_images, test_labels = maybe_limit_split(test_images, test_labels, args.limit_test)

    train_cache = load_or_create_cache("train", x_train, y_train, args.cache_dir, args=args, augment=True)
    test_cache = load_or_create_cache("test", test_images, test_labels, args.cache_dir, args=args, augment=False)

    raw_mean = train_cache["raw"].mean(axis=0, dtype=np.float64).astype(np.float32)
    raw_std = train_cache["raw"].std(axis=0, dtype=np.float64).astype(np.float32)
    hog_mean = train_cache["hog"].mean(axis=0, dtype=np.float64).astype(np.float32)
    hog_std = train_cache["hog"].std(axis=0, dtype=np.float64).astype(np.float32)

    raw_std = np.where(raw_std < 1e-6, 1.0, raw_std)
    hog_std = np.where(hog_std < 1e-6, 1.0, hog_std)

    train_ds = FeatureDataset(train_cache["raw"], train_cache["hog"], train_cache["labels"], raw_mean, raw_std, hog_mean, hog_std)
    test_ds = FeatureDataset(test_cache["raw"], test_cache["hog"], test_cache["labels"], raw_mean, raw_std, hog_mean, hog_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = MnistAnn(raw_dim=train_cache["raw"].shape[1], hog_dim=train_cache["hog"].shape[1], dropout=args.dropout).to(device)
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

    final_state = {key: value.cpu() for key, value in model.state_dict().items()}
    test_acc, confusion, per_class_accuracy = evaluate_test(model, test_loader, device)
    final_train_acc = history[-1]["train_acc"]
    final_train_loss = history[-1]["train_loss"]

    model_path = args.output_dir / "best_model.pt"
    metrics_path = args.output_dir / "metrics.json"
    confusion_path = args.output_dir / "confusion_matrix.npy"
    class_accuracy_path = args.output_dir / "class_accuracy.json"

    torch.save(
        {
            "model_state_dict": final_state,
            "raw_mean": raw_mean,
            "raw_std": raw_std,
            "hog_mean": hog_mean,
            "hog_std": hog_std,
            "args": vars(args),
            "final_train_acc": final_train_acc,
            "final_train_loss": final_train_loss,
            "test_acc": test_acc,
            "per_class_accuracy": per_class_accuracy,
        },
        model_path,
    )

    metrics = {
        "final_train_acc": final_train_acc,
        "final_train_loss": final_train_loss,
        "test_acc": test_acc,
        "per_class_accuracy": per_class_accuracy,
        "epochs_ran": len(history),
        "history": history,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    class_accuracy_path.write_text(json.dumps(per_class_accuracy, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(confusion_path, confusion)

    print(json.dumps({"final_train_acc": final_train_acc, "test_acc": test_acc}, ensure_ascii=False))
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {confusion_path}")
    print(f"Saved per-class accuracy to: {class_accuracy_path}")


if __name__ == "__main__":
    main()
