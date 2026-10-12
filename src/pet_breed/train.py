"""Train a pet-breed backbone with MLflow experiment tracking."""
import argparse
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from pet_breed.dataset import PetBreedDataset
from pet_breed.model import SUPPORTED_BACKBONES, create_model
from pet_breed.transforms import get_eval_transforms, get_train_transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, num_classes):
    training = optimizer is not None
    model.train(training)
    total_loss, predictions, targets = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=list(range(num_classes)), average="macro", zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--num-classes", type=int, default=37)
    parser.add_argument("--output", default=None)
    parser.add_argument("--experiment", default="pet-breed-classification")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, patience and lr must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(args.output or f"models/experiments/{args.backbone}_seed{args.seed}_lr{args.lr:g}_best.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_dataset = PetBreedDataset(args.manifest, split="train", transform=get_train_transforms())
    val_dataset = PetBreedDataset(args.manifest, split="val", transform=get_eval_transforms())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = create_model(num_classes=args.num_classes, pretrained=args.pretrained, backbone=args.backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name or f"{args.backbone}-lr{args.lr:g}-seed{args.seed}") as run:
        mlflow.log_params({
            "backbone": args.backbone, "learning_rate": args.lr, "batch_size": args.batch_size,
            "max_epochs": args.epochs, "patience": args.patience, "seed": args.seed,
            "pretrained": args.pretrained, "num_classes": args.num_classes,
            "optimizer": "AdamW", "device": str(device), "manifest": args.manifest,
        })
        best_f1, best_epoch, no_improvement = -1.0, 0, 0
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, train_loader, criterion, optimizer, device, args.num_classes)
            val = run_epoch(model, val_loader, criterion, None, device, args.num_classes)
            mlflow.log_metrics({
                "train_loss": train["loss"], "train_accuracy": train["accuracy"], "train_macro_f1": train["macro_f1"],
                "val_loss": val["loss"], "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
            }, step=epoch)
            print(f"Epoch {epoch}/{args.epochs} | train_loss={train['loss']:.4f} | val_acc={val['accuracy']:.4f} | val_f1={val['macro_f1']:.4f}", flush=True)
            if val["macro_f1"] > best_f1:
                best_f1, best_epoch, no_improvement = val["macro_f1"], epoch, 0
                torch.save({
                    "model_state_dict": model.state_dict(), "model_name": args.backbone,
                    "num_classes": args.num_classes, "epoch": epoch, "seed": args.seed,
                    "val_metrics": val,
                }, output_path)
            else:
                no_improvement += 1
            if no_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        mlflow.log_metrics({"best_val_macro_f1": best_f1, "best_epoch": best_epoch})
        mlflow.log_artifact(str(output_path), artifact_path="checkpoints")
        mlflow.set_tags({"phase": "training", "calibration_status": "pending", "evaluation_status": "pending"})
        print(f"Run ID: {run.info.run_id}\nBest checkpoint: {output_path}\nBest val macro-F1: {best_f1:.4f}")
        print("ECE and temperature are NOT computed here; run validation calibration separately.")


if __name__ == "__main__":
    main()
