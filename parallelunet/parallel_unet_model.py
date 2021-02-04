""" Full assembly of the parts to form the complete network """

from .parallel_unet_parts import *


class ParallelUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, cross_stitch_enable=True):
        super(ParallelUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.cross_stitch_enable = cross_stitch_enable

        self.inc = ParallelDoubleConv(n_channels, 64)
        self.down1 = ParallelDown(64, 128, cross_stitch_enable, alpha=0.1, beta=0.9)
        self.down2 = ParallelDown(128, 256, cross_stitch_enable, alpha=0.1, beta=0.9)
        self.down3 = ParallelDown(256, 512, cross_stitch_enable, alpha=0.9, beta=0.1)
        factor = 2 if bilinear else 1
        self.down4 = ParallelDown(512, 1024 // factor, cross_stitch_enable, alpha=0.9, beta=0.1)
        self.up1 = ParallelUp(1024, 512 // factor, bilinear)
        self.up2 = ParallelUp(512, 256 // factor, bilinear)
        self.up3 = ParallelUp(256, 128 // factor, bilinear)
        self.up4 = ParallelUp(128, 64, bilinear)
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
