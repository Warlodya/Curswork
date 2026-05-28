import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .datasets import FRAME_MEAN, FRAME_STD, HEATMAP_MEAN, HEATMAP_STD
from .model import TwoStreamLateFusionNet

CLASS_NAMES = ["Distracted", "Attentive"]


def _frame_transform(image_size: Tuple[int, int]):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(FRAME_MEAN, FRAME_STD),
        ]
    )


def _heatmap_transform(image_size: Tuple[int, int]):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(HEATMAP_MEAN, HEATMAP_STD),
        ]
    )


def load_trained_model(weights_path: str, device: str = None):
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(weights_path, map_location="cpu")
    image_size = tuple(ckpt.get("image_size", [224, 224]))

    model = TwoStreamLateFusionNet(pretrained_frame=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device_obj)
    model.eval()
    return model, CLASS_NAMES, image_size, device_obj


@torch.no_grad()
def predict_pair_pil(
    frame_pil: Image.Image,
    heatmap_pil: Image.Image,
    model: torch.nn.Module,
    class_names,
    image_size: Tuple[int, int],
    device: torch.device,
) -> Dict:
    frame_tensor = _frame_transform(image_size)(frame_pil.convert("RGB")).unsqueeze(0).to(device)
    heat_tensor = _heatmap_transform(image_size)(heatmap_pil.convert("L")).unsqueeze(0).to(device)

    logit = model(frame_tensor, heat_tensor)[0, 0]
    prob_attentive = float(torch.sigmoid(logit).item())
    pred_idx = 1 if prob_attentive >= 0.5 else 0

    return {
        "predicted_index": pred_idx,
        "predicted_label": class_names[pred_idx],
        "confidence": prob_attentive if pred_idx == 1 else 1.0 - prob_attentive,
        "probabilities": {
            "Distracted": 1.0 - prob_attentive,
            "Attentive": prob_attentive,
        },
    }


def predict_pair_paths(
    frame_path: str,
    heatmap_path: str,
    model: torch.nn.Module,
    class_names,
    image_size: Tuple[int, int],
    device: torch.device,
) -> Dict:
    frame_pil = Image.open(frame_path).convert("RGB")
    heatmap_pil = Image.open(heatmap_path).convert("L")
    result = predict_pair_pil(frame_pil, heatmap_pil, model, class_names, image_size, device)
    result["frame_path"] = str(Path(frame_path).resolve())
    result["heatmap_path"] = str(Path(heatmap_path).resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-stream inference on video frames and matching heatmaps."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="lecture.mp4",
        help="Path to lecture video file.",
    )
    parser.add_argument(
        "--heatmap_dir",
        type=str,
        default="heatmaps",
        help="Directory containing matching heatmap_XXXXXX.jpg files.",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="artifacts/two_stream/best_model.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="How many frames per second to process.",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="report.json",
        help="Path to output JSON analytics report.",
    )
    return parser.parse_args()


def _find_heatmap_path(heatmap_dir: Path, sample_idx: int) -> Path:
    # Expected naming: heatmap_000001.jpg, heatmap_000002.jpg, ...
    direct = heatmap_dir / f"heatmap_{sample_idx:06d}.jpg"
    if direct.exists():
        return direct

    # Fallback to any extension if JPG is not used.
    matches = sorted(heatmap_dir.glob(f"heatmap_{sample_idx:06d}.*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Missing heatmap for sample index {sample_idx}: {direct}")


def _frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def run_video_inference(
    video_path: str,
    heatmap_dir: str,
    weights_path: str,
    fps: float,
    report_path: str,
) -> Dict:
    if fps <= 0:
        raise ValueError("--fps must be > 0")

    model, class_names, image_size, device = load_trained_model(weights_path=weights_path)
    del class_names

    heatmap_root = Path(heatmap_dir)
    if not heatmap_root.exists():
        raise FileNotFoundError(f"Heatmap directory does not exist: {heatmap_root}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or np.isnan(source_fps):
        source_fps = 25.0
    frame_step = max(int(round(source_fps / fps)), 1)

    analytics: List[Dict] = []
    frame_idx = 0
    sample_idx = 1
    report_parent = Path(report_path).parent.resolve()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            timestamp_sec = int(round(frame_idx / source_fps))
            frame_pil = _frame_to_pil(frame_bgr)

            heatmap_path = _find_heatmap_path(heatmap_root, sample_idx)
            heatmap_pil = Image.open(heatmap_path).convert("L")

            pred = predict_pair_pil(
                frame_pil=frame_pil,
                heatmap_pil=heatmap_pil,
                model=model,
                class_names=CLASS_NAMES,
                image_size=image_size,
                device=device,
            )
            attention_score = int(round(pred["probabilities"]["Attentive"] * 100))
            try:
                heatmap_rel = heatmap_path.resolve().relative_to(report_parent)
                heatmap_file_value = str(heatmap_rel).replace("\\", "/")
            except ValueError:
                # Fallback if heatmap is outside the report directory tree.
                heatmap_file_value = str(heatmap_path).replace("\\", "/")

            analytics.append(
                {
                    "timestamp": timestamp_sec,
                    "attention_score": attention_score,
                    "heatmap_file": heatmap_file_value,
                }
            )
            sample_idx += 1

        frame_idx += 1

    cap.release()

    report = {
        "video_name": Path(video_path).name,
        "analytics": analytics,
    }

    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    args = parse_args()
    report = run_video_inference(
        video_path=args.video_path,
        heatmap_dir=args.heatmap_dir,
        weights_path=args.weights_path,
        fps=args.fps,
        report_path=args.report_path,
    )
    print(f"Saved report with {len(report['analytics'])} points to: {Path(args.report_path).resolve()}")


if __name__ == "__main__":
    main()
