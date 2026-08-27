"""model.py — SE-ResEuroNet architecture. Must match training exactly."""

import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        squeeze = self.pool(x).view(b, c)
        excite = self.fc(squeeze).view(b, c, 1, 1)
        return x * excite


class SEResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.se = SqueezeExcitation(out_channels)

        self.needs_projection = stride != 1 or in_channels != out_channels
        if self.needs_projection:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        out = self.se(out)
        if self.needs_projection:
            identity = self.projection(identity)
        return out + identity


class SEResEuroNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 10,
                 stage_channels=(64, 128, 256, 512), blocks_per_stage=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stage_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stage_channels[0]),
            nn.ReLU(inplace=True),
        )
        stages = []
        in_ch = stage_channels[0]
        for stage_idx, (out_ch, n_blocks) in enumerate(zip(stage_channels, blocks_per_stage)):
            stride = 1 if stage_idx == 0 else 2
            stages.append(SEResidualBlock(in_ch, out_ch, stride=stride))
            for _ in range(n_blocks - 1):
                stages.append(SEResidualBlock(out_ch, out_ch, stride=1))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(stage_channels[-1], num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)