import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames and generate paired synthetic gaze heatmaps."
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to lecture video file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/two_stream_attention",
        help="Output dataset root.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("H", "W"),
        help="Target frame/heatmap size.",
    )
    parser.add_argument(
        "--extract_fps",
        type=float,
        default=1.0,
        help="How many frames per second to extract from the video.",
    )
    parser.add_argument(
        "--copies_per_frame",
        type=int,
        default=1,
        help="How many synthetic heatmaps to generate per class for each extracted frame.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--max_frames", type=int, default=0, help="0 means use all extracted frames.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must be 1.0, got {ratio_sum:.5f}."
        )


def make_output_structure(root: Path) -> None:
    # Structure: split / class / frame_XXXXXX.jpg + heatmap_XXXXXX.jpg
    for split in ["train", "val", "test"]:
        for class_id in ["0", "1"]:
            (root / split / class_id).mkdir(parents=True, exist_ok=True)


def extract_frames_from_video(
    video_path: str,
    image_size: Tuple[int, int],
    extract_fps: float,
    max_frames: int = 0,
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or np.isnan(src_fps):
        src_fps = 25.0

    frame_step = max(int(round(src_fps / extract_fps)), 1)
    target_h, target_w = image_size
    frames: List[np.ndarray] = []

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)
            if max_frames > 0 and len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    if not frames:
        raise ValueError("No frames were extracted. Check the video path and extract_fps.")
    return frames


def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, Sequence[int]]:
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def random_point_distracted(height: int, width: int, margin_ratio: float = 0.2) -> Tuple[int, int]:
    # Distracted gaze is biased to edges / non-content regions.
    margin_h = max(int(height * margin_ratio), 1)
    margin_w = max(int(width * margin_ratio), 1)
    region = random.choice(["top", "bottom", "left", "right"])

    if region == "top":
        return random.randint(0, width - 1), random.randint(0, margin_h - 1)
    if region == "bottom":
        return random.randint(0, width - 1), random.randint(height - margin_h, height - 1)
    if region == "left":
        return random.randint(0, margin_w - 1), random.randint(0, height - 1)
    return random.randint(width - margin_w, width - 1), random.randint(0, height - 1)


def random_point_attentive(height: int, width: int) -> Tuple[int, int]:
    # Attentive gaze is biased to center/content area of lecture slides.
    x0, x1 = int(width * 0.2), int(width * 0.8)
    y0, y1 = int(height * 0.18), int(height * 0.75)
    return random.randint(x0, x1 - 1), random.randint(y0, y1 - 1)


def generate_heatmap(height: int, width: int, label: int) -> np.ndarray:
    heat = np.zeros((height, width), dtype=np.float32)
    num_blobs = random.randint(4, 9)

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    for _ in range(num_blobs):
        if label == 1:
            # Mostly attentive points, with a little noise.
            if random.random() < 0.85:
                cx, cy = random_point_attentive(height, width)
            else:
                cx, cy = random.randint(0, width - 1), random.randint(0, height - 1)
        else:
            # Mostly distracted points, with occasional center noise.
            if random.random() < 0.9:
                cx, cy = random_point_distracted(height, width)
            else:
                cx, cy = random_point_attentive(height, width)

        sigma = random.uniform(7.5, 20.0)
        amplitude = random.uniform(0.7, 1.0)
        heat += amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))

    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)
    return np.uint8(np.clip(heat * 255.0, 0, 255))


def generate_and_save_pairs(
    frames: List[np.ndarray],
    split_map: Dict[str, Sequence[int]],
    output_dir: Path,
    image_size: Tuple[int, int],
    copies_per_frame: int,
) -> None:
    h, w = image_size
    for split in ["train", "val", "test"]:
        for label in [0, 1]:
            sample_id = 1
            indices = split_map[split]
            desc = f"{split} class={label}"
            for frame_idx in tqdm(indices, desc=desc):
                frame = frames[frame_idx]
                for _ in range(copies_per_frame):
                    heatmap = generate_heatmap(h, w, label)
                    class_dir = output_dir / split / str(label)
                    frame_name = f"frame_{sample_id:06d}.jpg"
                    heatmap_name = f"heatmap_{sample_id:06d}.jpg"

                    cv2.imwrite(
                        str(class_dir / frame_name),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(str(class_dir / heatmap_name), heatmap)
                    sample_id += 1


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    validate_split_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    out_dir = Path(args.output_dir)
    make_output_structure(out_dir)

    image_size = tuple(args.image_size)
    frames = extract_frames_from_video(
        video_path=args.video_path,
        image_size=image_size,
        extract_fps=args.extract_fps,
        max_frames=args.max_frames,
    )
    split_map = split_indices(
        n=len(frames),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    generate_and_save_pairs(
        frames=frames,
        split_map=split_map,
        output_dir=out_dir,
        image_size=image_size,
        copies_per_frame=args.copies_per_frame,
    )

    print(f"Saved paired dataset to: {out_dir.resolve()}")
    print(
        "Class names: 0='Distracted', 1='Attentive'. "
        f"Extracted base frames: {len(frames)}"
    )


if __name__ == "__main__":
    main()

