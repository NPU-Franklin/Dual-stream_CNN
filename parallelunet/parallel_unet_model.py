""" Full assembly of the parts to form the complete network """

from .parallel_unet_parts import *


class ParallelUNet(nn.Module):
    def __init__(self, n_channels, n_classes, img_size, bilinear=True, bridge_enable=True):
        super(ParallelUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.bilinear = bilinear
        self.bridge_enable = bridge_enable

        self.inc = ParallelDoubleConv(n_channels, 64)
        self.down1 = ParallelDown(64, 128, bridge_enable)
        self.down2 = ParallelDown(128, 256, bridge_enable)
        self.down3 = ParallelDown(256, 512, bridge_enable)
        factor = 2 if bilinear else 1
        self.down4 = ParallelDown(512, 1024 // factor, bridge_enable)
        self.up1 = ParallelUp(1024, 512 // factor, bridge_enable, bilinear)
        self.up2 = ParallelUp(512, 256 // factor, bridge_enable, bilinear)
        self.up3 = ParallelUp(256, 128 // factor, bridge_enable, bilinear)
        self.up4 = ParallelUp(128, 64, bridge_enable, bilinear)
        self.outc = ParallelOutConv(64, n_classes)

    def forward(self, x):
        x1_1, x1_2 = self.inc(x, x)
        # down1
        x2_1, x2_2 = self.down1(x1_1, x1_2)
        # down2
        x3_1, x3_2 = self.down2(x2_1, x2_2)
        # down3
        x4_1, x4_2 = self.down3(x3_1, x3_2)
        # down4
        x5_1, x5_2 = self.down4(x4_1, x4_2)
        # up1
        x1, x2 = self.up1(x5_1, x4_1, x5_2, x4_2)
        # up2
        x1, x2 = self.up2(x1, x3_1, x2, x3_2)
        # up3
        x1, x2 = self.up3(x1, x2_1, x2, x2_2)
        # up4
        x1, x2 = self.up4(x1, x1_1, x2, x1_2)

        logits1, logits2 = self.outc(x1, x2)
        return logits1, logits2
