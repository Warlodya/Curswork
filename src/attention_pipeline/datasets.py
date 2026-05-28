from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


FRAME_MEAN = [0.485, 0.456, 0.406]
FRAME_STD = [0.229, 0.224, 0.225]
HEATMAP_MEAN = [0.5]
HEATMAP_STD = [0.25]

CLASS_NAMES = ["Distracted", "Attentive"]


def build_transforms(
    image_size: Tuple[int, int] = (224, 224),
    train: bool = True,
):
    frame_ops = [transforms.Resize(image_size)]
    if train:
        frame_ops.extend(
            [
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
            ]
        )
    frame_ops.extend([transforms.ToTensor(), transforms.Normalize(FRAME_MEAN, FRAME_STD)])
    frame_transform = transforms.Compose(frame_ops)

    heatmap_ops = [transforms.Resize(image_size)]
    heatmap_ops.extend([transforms.ToTensor(), transforms.Normalize(HEATMAP_MEAN, HEATMAP_STD)])
    heatmap_transform = transforms.Compose(heatmap_ops)
    return frame_transform, heatmap_transform


def _resolve_heatmap_pair(class_dir: Path, sample_suffix: str) -> Optional[Path]:
    matches = sorted(class_dir.glob(f"heatmap_{sample_suffix}.*"))
    if not matches:
        return None
    return matches[0]


def collect_paired_samples(split_dir: Path) -> List[Tuple[Path, Path, int]]:
    samples: List[Tuple[Path, Path, int]] = []
    for class_name in ["0", "1"]:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue

        label = int(class_name)
        frame_files = sorted(class_dir.glob("frame_*.*"))
        for frame_path in frame_files:
            # Suffix keeps zero-padded index, e.g. "000123" from "frame_000123.jpg"
            sample_suffix = frame_path.stem.replace("frame_", "", 1)
            heatmap_path = _resolve_heatmap_pair(class_dir, sample_suffix)
            if heatmap_path is None:
                continue
            samples.append((frame_path, heatmap_path, label))
    return samples


class AttentionTwoStreamDataset(Dataset):
    """
    Loads paired data:
      - RGB lecture frame: frame_XXXXXX.jpg
      - Grayscale heatmap: heatmap_XXXXXX.jpg
      - binary label from folder name (0/1)
    """

    def __init__(
        self,
        split_dir: str,
        frame_transform=None,
        heatmap_transform=None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.samples = collect_paired_samples(self.split_dir)
        if not self.samples:
            raise ValueError(f"No paired samples found in: {self.split_dir}")

        self.frame_transform = frame_transform
        self.heatmap_transform = heatmap_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        frame_path, heatmap_path, label = self.samples[idx]

        frame_img = Image.open(frame_path).convert("RGB")
        heatmap_img = Image.open(heatmap_path).convert("L")

        if self.frame_transform is not None:
            frame_tensor = self.frame_transform(frame_img)
        else:
            frame_tensor = transforms.ToTensor()(frame_img)

        if self.heatmap_transform is not None:
            heatmap_tensor = self.heatmap_transform(heatmap_img)
        else:
            heatmap_tensor = transforms.ToTensor()(heatmap_img)

        return {
            "frame": frame_tensor,
            "heatmap": heatmap_tensor,
            "label": float(label),
            "frame_path": str(frame_path),
            "heatmap_path": str(heatmap_path),
        }


def create_dataloaders(
    data_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, Sequence[str], Dict[str, int]]:
    root = Path(data_dir)

    train_frame_tf, train_heat_tf = build_transforms(image_size=image_size, train=True)
    eval_frame_tf, eval_heat_tf = build_transforms(image_size=image_size, train=False)

    train_ds = AttentionTwoStreamDataset(
        split_dir=str(root / "train"),
        frame_transform=train_frame_tf,
        heatmap_transform=train_heat_tf,
    )
    val_ds = AttentionTwoStreamDataset(
        split_dir=str(root / "val"),
        frame_transform=eval_frame_tf,
        heatmap_transform=eval_heat_tf,
    )
    test_ds = AttentionTwoStreamDataset(
        split_dir=str(root / "test"),
        frame_transform=eval_frame_tf,
        heatmap_transform=eval_heat_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    label_map = {"Distracted": 0, "Attentive": 1}
    return train_loader, val_loader, test_loader, CLASS_NAMES, label_map
