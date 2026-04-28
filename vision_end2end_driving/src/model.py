import torch
from torch import nn


class EndToEndDrivingNet(nn.Module):
    """Small DAVE-2 inspired CNN for image-to-control regression."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 100),
            nn.ELU(),
            nn.Dropout(0.15),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.features(x))
        steering = torch.tanh(raw[:, 0:1])
        throttle_brake = torch.sigmoid(raw[:, 1:3])
        return torch.cat([steering, throttle_brake], dim=1)
