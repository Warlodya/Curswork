import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training history curves.")
    parser.add_argument(
        "--history_path",
        type=str,
        default="artifacts/two_stream/training_history.json",
        help="Path to training_history.json produced by train.py",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="artifacts/two_stream/training_history.png",
        help="Where to save the output figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving.",
    )
    return parser.parse_args()


def _get_series(history: Dict, primary_key: str, fallback_key: str) -> List[float]:
    if primary_key in history and isinstance(history[primary_key], list):
        return history[primary_key]
    if fallback_key in history and isinstance(history[fallback_key], list):
        return history[fallback_key]
    return []


def load_history(history_path: Path) -> Tuple[List[float], List[float], List[float], List[float]]:
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    train_loss = _get_series(history, "train_loss", "loss")
    val_loss = _get_series(history, "val_loss", "validation_loss")
    train_acc = _get_series(history, "train_acc", "accuracy")
    val_acc = _get_series(history, "val_acc", "validation_accuracy")

    if not train_loss and not val_loss and not train_acc and not val_acc:
        raise ValueError(
            "No recognized metric arrays found in history JSON. "
            "Expected keys like train_loss, val_loss, train_acc, val_acc."
        )

    return train_loss, val_loss, train_acc, val_acc


def plot_history(
    train_loss: List[float],
    val_loss: List[float],
    train_acc: List[float],
    val_acc: List[float],
    output_path: Path,
    show: bool = False,
) -> None:
    max_len = max(len(train_loss), len(val_loss), len(train_acc), len(val_acc))
    epochs = list(range(1, max_len + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Loss subplot
    if train_loss:
        axes[0].plot(epochs[: len(train_loss)], train_loss, label="Train Loss", linewidth=2)
    if val_loss:
        axes[0].plot(epochs[: len(val_loss)], val_loss, label="Val Loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    # Accuracy subplot
    if train_acc:
        axes[1].plot(epochs[: len(train_acc)], train_acc, label="Train Accuracy", linewidth=2)
    if val_acc:
        axes[1].plot(epochs[: len(val_acc)], val_acc, label="Val Accuracy", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("Training History", fontsize=13)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    history_path = Path(args.history_path)
    output_path = Path(args.output_path)

    train_loss, val_loss, train_acc, val_acc = load_history(history_path)
    plot_history(
        train_loss=train_loss,
        val_loss=val_loss,
        train_acc=train_acc,
        val_acc=val_acc,
        output_path=output_path,
        show=args.show,
    )
    print(f"Saved training history plot: {output_path.resolve()}")


if __name__ == "__main__":
    main()
