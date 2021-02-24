"""
cross_stitch unit for dual_stream CNN
"""
import torch
import torch.nn as nn


class CrossStitch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cross_stitch = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input1, input2):
        input1 = input1 * self.cross_stitch[0]
        input2 = input2 * self.cross_stitch[1]
        output = torch.cat([input1, input2], dim=1)

        return self.conv(output)
