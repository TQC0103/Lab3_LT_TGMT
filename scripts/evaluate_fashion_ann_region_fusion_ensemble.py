from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_fashion_ann_region_fusion import (
    FeatureDataset,
    SingleBranchAnn,
    ensure_dir,
    load_dataset_arrays,
    load_or_create_cache,
    maybe_limit_split,
)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def predict_ensemble(
    checkpoint_paths: list[Path],
    data_root: Path,
    cache_root: Path,
    batch_size: int,
    num_workers: int,
    limit_test: int | None,
) -> tuple[float, np.ndarray, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_images, test_labels = load_dataset_arrays(data_root)
    test_images, test_labels = maybe_limit_split(test_images, test_labels, limit_test)

    checkpoints = [load_checkpoint(path) for path in checkpoint_paths]
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    probs_accumulator: np.ndarray | None = None

    for checkpoint in checkpoints:
        args_dict = checkpoint.get("args", {})
        cache_dir = cache_root / f"seed_{args_dict.get('seed', 0)}"
        ensure_dir(cache_dir)
        namespace = argparse.Namespace(**args_dict)
        test_cache = load_or_create_cache("test", test_images, test_labels, cache_dir, args=namespace, augment=False)

        mean = checkpoint["feature_mean"]
        std = checkpoint["feature_std"]
        std = np.where(std < 1e-6, 1.0, std)

        test_ds = FeatureDataset(test_cache["features"], test_cache["labels"], mean, std)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

        model = SingleBranchAnn(input_dim=test_cache["features"].shape[1], dropout=float(args_dict.get("dropout", 0.25)))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        all_probs = []
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(device)
                logits = model(features)
                all_probs.append(F.softmax(logits, dim=1).cpu().numpy())

        probs = np.concatenate(all_probs, axis=0)
        probs_accumulator = probs if probs_accumulator is None else probs_accumulator + probs

    assert probs_accumulator is not None
    mean_probs = probs_accumulator / len(checkpoints)
    preds = mean_probs.argmax(axis=1)

    confusion = np.zeros((10, 10), dtype=np.int64)
    for truth, pred in zip(test_labels, preds, strict=True):
        confusion[int(truth), int(pred)] += 1

    accuracy = float((preds == test_labels).mean())
    per_class_accuracy: dict[str, float] = {}
    for class_idx in range(10):
        total_in_class = confusion[class_idx].sum()
        per_class_accuracy[str(class_idx)] = float(confusion[class_idx, class_idx] / total_in_class) if total_in_class > 0 else 0.0

    return accuracy, confusion, per_class_accuracy, test_labels.astype(np.int64), preds.astype(np.int64), mean_probs.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of FashionMNIST region-fusion ANN checkpoints.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/fashion_mnist_region_fusion_ensemble_eval_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fashion_mnist_ann_region_fusion_ensemble"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()

    ensure_dir(args.cache_root)
    ensure_dir(args.output_dir)

    accuracy, confusion, per_class_accuracy, targets, preds, probs = predict_ensemble(
        checkpoint_paths=args.checkpoints,
        data_root=args.data_root,
        cache_root=args.cache_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_test=args.limit_test,
    )

    metrics = {
        "test_acc": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "num_models": len(args.checkpoints),
        "checkpoints": [str(path) for path in args.checkpoints],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "class_accuracy.json").write_text(json.dumps(per_class_accuracy, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(args.output_dir / "confusion_matrix.npy", confusion)
    np.save(args.output_dir / "test_targets.npy", targets)
    np.save(args.output_dir / "test_predictions.npy", preds)
    np.save(args.output_dir / "test_probabilities.npy", probs)

    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
