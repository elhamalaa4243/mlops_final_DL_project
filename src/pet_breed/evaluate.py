"""Evaluate a frozen pet-breed classifier on the untouched test split.

Example:
    python -m pet_breed.evaluate \
      --checkpoint models/frozen/resnet18_v1.pt \
      --batch-size 32
"""

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from pet_breed.dataset import PetBreedDataset
from pet_breed.model import create_model
from pet_breed.transforms import get_eval_transforms


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_calibration_error(confidences, correct, bins=15):
    """Equal-width-bin ECE, using the top predicted class confidence."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidences)
    ece = 0.0
    details = []
    for i in range(bins):
        low, high = float(edges[i]), float(edges[i + 1])
        mask = (confidences >= low) & (confidences < high)
        if i == bins - 1:
            mask |= confidences == high
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidences[mask].mean())
            bin_accuracy = float(correct[mask].mean())
            ece += count / total * abs(bin_accuracy - mean_confidence)
        else:
            mean_confidence = None
            bin_accuracy = None
        details.append({
            "lower": low,
            "upper": high,
            "count": count,
            "accuracy": bin_accuracy,
            "confidence": mean_confidence,
        })
    return float(ece), details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/frozen/resnet18_v1.pt")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--classes", default="data/processed/classes.json")
    parser.add_argument("--output-dir", default="reports/resnet18_v1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.batch_size < 1 or args.ece_bins < 1 or args.num_workers < 0:
        parser.error("batch-size and ece-bins must be positive; num-workers must be nonnegative")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    with Path(args.classes).open(encoding="utf-8") as stream:
        classes = json.load(stream)
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        raise ValueError("classes.json must be a list of breed names")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    num_classes = int(checkpoint["num_classes"])
    if len(classes) != num_classes:
        raise ValueError(f"Checkpoint has {num_classes} classes but classes.json has {len(classes)}")

    model_name = checkpoint.get("model_name", "resnet18")
    if model_name != "resnet18":
        raise ValueError(f"This evaluator currently supports resnet18 only; got {model_name!r}")

    model = create_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    dataset = PetBreedDataset(args.manifest, split="test", transform=get_eval_transforms())
    if len(dataset) == 0:
        raise ValueError("The test split is empty")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    predictions, targets, confidences = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            predictions.extend(pred.cpu().tolist())
            targets.extend(labels.tolist())
            confidences.extend(conf.cpu().tolist())

    y_true = np.asarray(targets, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    confidence = np.asarray(confidences, dtype=np.float64)
    correct = y_pred == y_true
    labels = list(range(num_classes))

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    ece, reliability = expected_calibration_error(confidence, correct, bins=args.ece_bins)
    per_class = classification_report(
        y_true, y_pred, labels=labels, target_names=classes,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    thresholds = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]
    selective_metrics = []
    for threshold in thresholds:
        accepted = confidence >= threshold
        accepted_count = int(accepted.sum())
        selective_metrics.append({
            "threshold": threshold,
            "accepted": accepted_count,
            "coverage": accepted_count / len(y_true),
            "selective_accuracy": float(correct[accepted].mean()) if accepted_count else None,
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "model_version": "resnet18_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_training_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_metrics": checkpoint.get("val_metrics"),
        "test_samples": len(y_true),
        "num_classes": num_classes,
        "device": device,
        "hardware": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or platform.machine(),
        "batch_size": args.batch_size,
        "torch_version": torch.__version__,
        "preprocessing": "pet_breed.transforms.get_eval_transforms",
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_ece": ece,
        "ece_bins": args.ece_bins,
        "selective_metrics": selective_metrics,
    }

    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "classification_report.json").write_text(json.dumps(per_class, indent=2), encoding="utf-8")
    (output_dir / "reliability_bins.json").write_text(json.dumps(reliability, indent=2), encoding="utf-8")
    np.save(output_dir / "confusion_matrix.npy", cm)
    (output_dir / "confusion_matrix.csv").write_text(
        ",".join(["true/pred"] + classes) + "\n" +
        "\n".join(classes[i] + "," + ",".join(map(str, row)) for i, row in enumerate(cm)) + "\n",
        encoding="utf-8",
    )

    print(f"Model: {model_name} | checkpoint epoch: {checkpoint.get('epoch')} | device: {device}")
    print(f"Test samples: {len(y_true)}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"Test macro-F1: {macro_f1:.4f}")
    print(f"Test ECE ({args.ece_bins} bins): {ece:.4f}")
    print("Coverage / selective accuracy:")
    for item in selective_metrics:
        acc = "N/A" if item["selective_accuracy"] is None else f'{item["selective_accuracy"]:.2%}'
        print(f'  confidence >= {item["threshold"]:.2f}: coverage={item["coverage"]:.2%}, accuracy={acc}')
    print(f"Reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
