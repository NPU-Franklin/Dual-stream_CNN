"""
bridge unit for parallel-network
"""
import torch
import torch.nn as nn


class Bridge(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # initialize
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input1, input2):
        output = torch.cat([input1, input2], dim=1)

        return self.conv(output)
