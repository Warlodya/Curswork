import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from .datasets import create_dataloaders
from .model import EarlyStopping, TwoStreamLateFusionNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Two-Stream CNN (Late Fusion).")
    parser.add_argument("--data_dir", type=str, required=True, help="Root of generated dataset.")
    parser.add_argument("--output_dir", type=str, default="artifacts/two_stream")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], metavar=("H", "W"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_pretrained_frame", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binary_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == labels).float().sum().item()
    return correct / max(labels.size(0), 1)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: bool = True,
) -> Tuple[float, float]:
    model.train(mode=training)
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, leave=False):
            frame = batch["frame"].to(device, non_blocking=True)
            heatmap = batch["heatmap"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float().view(-1, 1)

            logits = model(frame, heatmap)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_acc += binary_accuracy_from_logits(logits, labels) * batch_size
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_acc / max(total_samples, 1)
    return avg_loss, avg_acc


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names,
    image_size,
    epoch: int,
) -> None:
    payload: Dict = {
        "model_state_dict": model.state_dict(),
        "class_names": list(class_names),
        "image_size": list(image_size),
        "epoch": epoch,
        "model_type": "TwoStreamLateFusionNet",
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = tuple(args.image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, class_names, _ = create_dataloaders(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = TwoStreamLateFusionNet(
        dropout=args.dropout,
        pretrained_frame=not args.no_pretrained_frame,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=False,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        improved = early_stopping.step(val_loss)
        if improved:
            save_checkpoint(best_path, model, class_names, image_size, epoch)
            print(f"  Saved best model -> {best_path}")

        save_checkpoint(last_path, model, class_names, image_size, epoch)

        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch}.")
            break

    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Training complete.")
    print(f"Best model: {best_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
