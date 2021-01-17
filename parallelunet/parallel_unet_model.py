""" Full assembly of the parts to form the complete network """

from .bridge import *
from .parallel_unet_parts import *


class ParallelUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bridge_enable=True):
        super(ParallelUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.bridge_enable = bridge_enable

        self.inc = DoubleConv(n_channels, 64)
        self.MaxPool = nn.MaxPool2d(2)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.factor = 2 if bilinear else 1
        self.down4 = DoubleConv(512, 1024 // self.factor)
        self.upsample = nn.Upsample(scale_factor=2, mode='bililinear', align_corners=True)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1_1 = self.inc(x)
        x1_2 = self.inc(x)

        # down1
        x2_1 = self.MaxPool(x1_1)
        x2_2 = self.MaxPool(x1_2)

        if self.bridge_enable:
            x2_1, x2_2 = Bridge(x2_1, x2_2)

        x2_1 = self.down1(x2_1)
        x2_2 = self.down1(x2_2)

        # down2
        x3_1 = self.MaxPool(x2_1)
        x3_2 = self.MaxPool(x2_2)

        if self.bridge_enable:
            x3_1, x3_2 = Bridge(x3_1, x3_2)

        x3_1 = self.down2(x3_1)
        x3_2 = self.down2(x3_2)

        # down3
        x4_1 = self.MaxPool(x3_1)
        x4_2 = self.MaxPool(x3_2)

        if self.bridge_enable:
            x4_1, x4_2 = Bridge(x4_1, x4_2)

        x4_1 = self.down3(x4_1)
        x4_2 = self.down3(x4_2)

        # down4
        x5_1 = self.MaxPool(x4_1)
        x5_2 = self.MaxPool(x4_2)

        if self.bridge_enable:
            x5_1, x5_2 = Bridge(x5_1, x5_2)

        x5_1 = self.down4(x5_1)
        x5_2 = self.down4(x5_2)

        # up1
        if self.bilinear:
            x1 = self.upsample(x5_1)
            x2 = self.upsample(x5_2)
            x1 = copy_crop(x1, x4_1)
            x2 = copy_crop(x2, x4_2)
            if self.bridge_enable:
                x1, x2 = Bridge(x1, x2)
            x1 = DoubleConv(1024, 512 // self.factor, 1024 // 2)(x1)
            x2 = DoubleConv(1024, 512 // self.factor, 1024 // 2)(x2)
        else:
            x1 = nn.ConvTranspose2d(1024, 1024 // 2, kernel_size=2, stride=2)(x5_1)
            x2 = nn.ConvTranspose2d(1024, 1024 // 2, kernel_size=2, stride=2)(x5_2)
            x1 = copy_crop(x1, x4_1)
            x2 = copy_crop(x2, x4_2)
            if self.bridge_enable:
                x1, x2 = Bridge(x1, x2)
            x1 = DoubleConv(1024, 512 // self.factor)(x1)
            x2 = DoubleConv(1024, 512 // self.factor)(x2)

        # up2
        if self.bilinear:
            x1 = self.upsample(x1)
            x2 = self.upsample(x2)
            x1 = copy_crop(x1, x3_1)
            x2 = copy_crop(x2, x3_2)
            if self.bridge_enable:
                x1, x2 = Bridge(x1, x2)
            x1 = DoubleConv(512, 256 // self.factor, 512 // 2)(x1)
            x2 = DoubleConv(512, 256 // self.factor, 512 // 2)(x2)
        else:
            x1 = nn.ConvTranspose2d(512, 512 // 2, kernel_size=2, stride=2)(x1)
            x2 = nn.ConvTranspose2d(512, 512 // 2, kernel_size=2, stride=2)(x2)
            x1 = copy_crop(x1, x3_1)
            x2 = copy_crop(x2, x3_2)
            if self.bridge_enable:
                x1, x2 = Bridge(x1, x2)
            x1 = DoubleConv(512, 256 // self.factor)(x1)
            x2 = DoubleConv(512, 256 // self.factor)(x2)

        # up3
        if self.bilinear:
            x1 = self.upsample(x1)
            x2 = self.upsample(x2)
            x1 = copy_crop(x1, x2_1)
            x2 = copy_crop(x2, x2_2)
            if self.bridge_enable:
                x1, x2 = Bridge(x1, x2)
            x1 = DoubleConv(256, 128 // self.factor, 256 // 2)(x1)
            x2 = DoubleConv(256, 128 // self.factor, 256 // 2)(x2)
        else:
            x1 = nn.ConvTranspose2d(256, 256 // 2, kernel_size=2, stride=2)(x1)
            x2 = nn.ConvTranspose2d(256, 256 // 2, kernel_size=2, stride=2)(x2)
            x1 = copy_crop(x1, x2_1)
            x2 = copy_crop(x2, x2_2)
            if self.bridge_enable:
                x1, x2 = Bridge(x1, x2)
            x1 = DoubleConv(256, 128 // self.factor)(x1)
            x2 = DoubleConv(256, 128 // self.factor)(x2)

        # up4
        if self.bilinear:
            x1 = self.upsample(x1)
            x2 = self.upsample(x2)
            x1 = copy_crop(x1, x1_1)
            x2 = copy_crop(x2, x1_2)
            if self.bridge_enable:
                x1, x2 = Bridge(x1, x2)
            x1 = DoubleConv(128, 64, 128 // 2)(x1)
            x2 = DoubleConv(128, 64, 128 // 2)(x2)
        else:
            x1 = nn.ConvTranspose2d(128, 128 // 2, kernel_size=2, stride=2)(x1)
            x2 = nn.ConvTranspose2d(128, 128 // 2, kernel_size=2, stride=2)(x2)
            x1 = copy_crop(x1, x1_1)
            x2 = copy_crop(x2, x1_2)
            if self.bridge_enable:
                x1, x2 = Bridge(x1, x2)
            x1 = DoubleConv(128, 64)(x1)
            x2 = DoubleConv(128, 64)(x2)

        logits1 = self.outc(x1)
        logits2 = self.outc(x2)
        return logits1, logits2
