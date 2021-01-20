"""
Parts of the paralllelell U-Net mddel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_stitch import CrossStitch


class ParallelDoubleConv(nn.Module):
    """((convolution => [BN] => ReLU) * 2) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        return self.double_conv1(x1), self.double_conv2(x2)


class ParallelDown(nn.Module):
    """Downscaling with maxpool then bridging and double conv"""

    def __init__(self, in_channels, out_channels, cross_stitch_enable=True):
        super().__init__()
        self.cross_stitch_enable = cross_stitch_enable

        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)

        if cross_stitch_enable:
            self.cross_stitch = CrossStitch()

        self.conv = ParallelDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.maxpool1(x1)
        x2 = self.maxpool2(x2)

        if self.cross_stitch_enable:
            x1, x2 = self.cross_stitch(x1, x2)

        return self.conv(x1, x2)


class ParallelUp(nn.Module):
    """Upscaling then bridging and double conv"""

    def __init__(self, in_channels, out_channels, cross_stitch_enable=True, bilinear=True):
        super().__init__()
        self.cross_stitch_enable = cross_stitch_enable

        if bilinear:
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            if cross_stitch_enable:
                self.cross_stitch = CrossStitch()

            self.conv = ParallelDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

            if cross_stitch_enable:
                self.cross_stitch = CrossStitch()

            self.conv = ParallelDoubleConv(in_channels, out_channels)

    def forward(self, x1_1, x1_2, x2_1, x2_2):
        x1_1 = self.up1(x1_1)
        x2_1 = self.up2(x2_1)

        diffY_1 = x1_2.size()[2] - x1_1.size()[2]
        diffX_1 = x1_2.size()[3] - x1_1.size()[3]
        diffY_2 = x2_2.size()[2] - x2_1.size()[2]
        diffX_2 = x2_2.size()[3] - x2_1.size()[3]

        x1_1 = F.pad(x1_1, [diffX_1 // 2, diffX_1 - diffX_1 // 2,
                            diffY_1 // 2, diffY_1 - diffY_1 // 2])
        x2_1 = F.pad(x2_1, [diffX_2 // 2, diffX_2 - diffX_2 // 2,
                            diffY_2 // 2, diffY_2 - diffY_2 // 2])

        x1 = torch.cat([x1_1, x1_2], dim=1)
        x2 = torch.cat([x2_1, x2_2], dim=1)

        if self.cross_stitch_enable:
            x1, x2 = self.cross_stitch(x1, x2)

        return self.conv(x1, x2)


class ParallelOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ParallelOutConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        return self.conv1(x1), self.conv2(x2)
