from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class FrameBranch(nn.Module):
    """
    RGB frame branch based on MobileNetV2 feature extractor.
    """

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.proj(x)
        return x


class HeatmapBranch(nn.Module):
    """
    Grayscale heatmap branch with custom CNN.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        # Explicit last conv for Grad-CAM targeting.
        self.last_conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.last_bn = nn.BatchNorm2d(128)
        self.last_relu = nn.ReLU(inplace=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.last_conv(x)
        x = self.last_bn(x)
        x = self.last_relu(x)
        x = self.pool(x)
        x = self.proj(x)
        return x


class TwoStreamLateFusionNet(nn.Module):
    """
    Two-stream late-fusion model:
      - frame branch (RGB context)
      - heatmap branch (attention distribution)
      - concatenation + MLP fusion -> binary logit
    """

    def __init__(
        self,
        frame_embedding_dim: int = 256,
        heatmap_embedding_dim: int = 128,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.35,
        pretrained_frame: bool = True,
    ) -> None:
        super().__init__()
        self.frame_branch = FrameBranch(
            embedding_dim=frame_embedding_dim,
            pretrained=pretrained_frame,
        )
        self.heatmap_branch = HeatmapBranch(embedding_dim=heatmap_embedding_dim)

        fusion_dim = frame_embedding_dim + heatmap_embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, frame_x: torch.Tensor, heatmap_x: torch.Tensor) -> torch.Tensor:
        frame_feat = self.frame_branch(frame_x)
        heat_feat = self.heatmap_branch(heatmap_x)
        fused = torch.cat([frame_feat, heat_feat], dim=1)
        logit = self.classifier(fused)
        return logit


class EarlyStopping:
    """
    Stop training if validation loss does not improve for `patience` epochs.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False

