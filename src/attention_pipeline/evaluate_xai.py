import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from .datasets import CLASS_NAMES, create_dataloaders
from .model import TwoStreamLateFusionNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Two-Stream model and generate Grad-CAM.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts/two_stream_eval")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_gradcam_samples", type=int, default=16)
    return parser.parse_args()


def denormalize_heatmap_tensor(heatmap_tensor: torch.Tensor) -> np.ndarray:
    # Inverse normalization for visualization where original transform used mean=0.5, std=0.25.
    heat = heatmap_tensor.detach().cpu().numpy()[0]
    heat = (heat * 0.25) + 0.5
    heat = np.clip(heat, 0.0, 1.0)
    return np.uint8(heat * 255)


class HeatmapBranchGradCAM:
    """
    Grad-CAM for the last conv layer in the heatmap branch.
    """

    def __init__(self, model: TwoStreamLateFusionNet):
        self.model = model
        self.target_layer = model.heatmap_branch.last_conv
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd_hook(_, __, output):
            self.activations = output

        def bwd_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(
        self,
        frame_tensor: torch.Tensor,
        heatmap_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        self.model.zero_grad()
        logit = self.model(frame_tensor, heatmap_tensor)  # [1, 1]
        prob = torch.sigmoid(logit)[0, 0].item()

        if target_class is None:
            target_class = 1 if prob >= 0.5 else 0

        # For binary logits:
        # class=1 score -> +logit, class=0 score -> -logit
        score = logit[:, 0] if target_class == 1 else -logit[:, 0]
        score.backward(retain_graph=True)

        grads = self.gradients[0]         # [C, H, W]
        acts = self.activations[0]        # [C, H, W]
        weights = grads.mean(dim=(1, 2))  # [C]

        cam = torch.zeros(acts.shape[1:], device=acts.device)
        for c, w in enumerate(weights):
            cam += w * acts[c]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def evaluate_model(model, loader, device):
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            frame = batch["frame"].to(device, non_blocking=True)
            heatmap = batch["heatmap"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(frame, heatmap).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            y_true.extend(labels.cpu().numpy().astype(int).tolist())
            y_pred.extend(preds.cpu().numpy().astype(int).tolist())

    report_text = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, digits=4
    )
    report_dict = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, digits=4, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return report_text, report_dict, cm


def save_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_gradcam_visualizations(
    model: TwoStreamLateFusionNet,
    loader,
    device: torch.device,
    output_dir: Path,
    max_samples: int = 16,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gradcam = HeatmapBranchGradCAM(model)
    model.eval()

    saved = 0
    for batch in loader:
        frames = batch["frame"]
        heatmaps = batch["heatmap"]
        labels = batch["label"].numpy().astype(int).tolist()

        for i in range(frames.size(0)):
            if saved >= max_samples:
                return

            frame_tensor = frames[i : i + 1].to(device)
            heatmap_tensor = heatmaps[i : i + 1].to(device)

            with torch.no_grad():
                logit = model(frame_tensor, heatmap_tensor)[0, 0]
                prob = torch.sigmoid(logit).item()
                pred = 1 if prob >= 0.5 else 0

            cam = gradcam.generate(frame_tensor, heatmap_tensor, target_class=pred)

            heat_u8 = denormalize_heatmap_tensor(heatmaps[i])
            cam_resized = cv2.resize(cam, (heat_u8.shape[1], heat_u8.shape[0]))
            cam_u8 = np.uint8(np.clip(cam_resized * 255, 0, 255))

            heat_rgb = cv2.cvtColor(heat_u8, cv2.COLOR_GRAY2RGB)
            cam_color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
            cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(heat_rgb, 0.55, cam_color, 0.45, 0)

            side_by_side = np.concatenate([heat_rgb, overlay], axis=1)
            name = (
                f"sample_{saved:03d}_true-{CLASS_NAMES[labels[i]]}"
                f"_pred-{CLASS_NAMES[pred]}_p-{prob:.3f}.png"
            )
            cv2.imwrite(str(output_dir / name), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
            saved += 1


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.weights_path, map_location="cpu")
    image_size = tuple(checkpoint.get("image_size", [224, 224]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoStreamLateFusionNet(pretrained_frame=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    _, _, test_loader, _, _ = create_dataloaders(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    report_text, report_dict, cm = evaluate_model(model, test_loader, device)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    np.savetxt(output_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    save_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    save_gradcam_visualizations(
        model=model,
        loader=test_loader,
        device=device,
        output_dir=output_dir / "gradcam_heatmap_branch",
        max_samples=args.num_gradcam_samples,
    )

    print("Evaluation complete.")
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

