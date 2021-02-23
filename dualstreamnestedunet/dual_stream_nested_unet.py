import torch
import torch.nn as nn
from torch.nn import init

from crossstitch import CrossStitch


# initialize the module
def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class UnetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(UnetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(UnetUp, self).__init__()
        self.conv = UnetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv2') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


class DualStreamNestedUNet(nn.Module):

    def __init__(self, n_channels=1, n_classes=2, feature_scale=1, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(DualStreamNestedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.convx00 = UnetConv2(self.n_channels, filters[0], self.is_batchnorm)
        self.convx10 = UnetConv2(filters[0], filters[1], self.is_batchnorm)
        self.convx20 = UnetConv2(filters[1], filters[2], self.is_batchnorm)
        self.convx30 = UnetConv2(filters[2], filters[3], self.is_batchnorm)
        self.convx40 = UnetConv2(filters[3], filters[4], self.is_batchnorm)

        self.convy00 = UnetConv2(self.n_channels, filters[0], self.is_batchnorm)
        self.convy10 = UnetConv2(filters[0], filters[1], self.is_batchnorm)
        self.convy20 = UnetConv2(filters[1], filters[2], self.is_batchnorm)
        self.convy30 = UnetConv2(filters[2], filters[3], self.is_batchnorm)
        self.convy40 = UnetConv2(filters[3], filters[4], self.is_batchnorm)

        self.cross_stitch0 = CrossStitch(filters[0])
        self.cross_stitch1 = CrossStitch(filters[1])
        self.cross_stitch2 = CrossStitch(filters[2])
        self.cross_stitch3 = CrossStitch(filters[3])

        # upsampling
        self.up_concatx01 = UnetUp(filters[1], filters[0], self.is_deconv)
        self.up_concatx11 = UnetUp(filters[2], filters[1], self.is_deconv)
        self.up_concatx21 = UnetUp(filters[3], filters[2], self.is_deconv)
        self.up_concatx31 = UnetUp(filters[4], filters[3], self.is_deconv)

        self.up_concatx02 = UnetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concatx12 = UnetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concatx22 = UnetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concatx03 = UnetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concatx13 = UnetUp(filters[2], filters[1], self.is_deconv, 4)

        self.up_concatx04 = UnetUp(filters[1], filters[0], self.is_deconv, 5)

        self.up_concaty01 = UnetUp(filters[1], filters[0], self.is_deconv)
        self.up_concaty11 = UnetUp(filters[2], filters[1], self.is_deconv)
        self.up_concaty21 = UnetUp(filters[3], filters[2], self.is_deconv)
        self.up_concaty31 = UnetUp(filters[4], filters[3], self.is_deconv)

        self.up_concaty02 = UnetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concaty12 = UnetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concaty22 = UnetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concaty03 = UnetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concaty13 = UnetUp(filters[2], filters[1], self.is_deconv, 4)

        self.up_concaty04 = UnetUp(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.finalx_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.finalx_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.finalx_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.finalx_4 = nn.Conv2d(filters[0], n_classes, 1)

        self.finaly_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.finaly_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.finaly_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.finaly_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        x_00 = self.convx00(inputs)  # 16*512*512
        y_00 = self.convy00(inputs)
        maxpoolx0 = self.maxpool(x_00)  # 16*256*256
        maxpooly0 = self.maxpool(y_00)
        maxpool0 = self.cross_stitch0(maxpoolx0, maxpooly0)

        x_10 = self.convx10(maxpool0)  # 32*256*256
        y_10 = self.convy10(maxpool0)
        maxpoolx1 = self.maxpool(x_10)  # 32*128*128
        maxpooly1 = self.maxpool(y_10)
        maxpool1 = self.cross_stitch1(maxpoolx1, maxpooly1)

        x_20 = self.convx20(maxpool1)  # 64*128*128
        y_20 = self.convy20(maxpool1)
        maxpoolx2 = self.maxpool(x_20)  # 64*64*64
        maxpooly2 = self.maxpool(y_20)
        maxpool2 = self.cross_stitch2(maxpoolx2, maxpooly2)

        x_30 = self.convx30(maxpool2)  # 128*64*64
        y_30 = self.convy30(maxpool2)
        maxpoolx3 = self.maxpool(x_30)  # 128*32*32
        maxpooly3 = self.maxpool(y_30)
        maxpool3 = self.cross_stitch3(maxpoolx3, maxpooly3)

        x_40 = self.convx40(maxpool3)  # 256*32*32
        y_40 = self.convy40(maxpool3)

        # column : 1
        x_01 = self.up_concatx01(x_10, x_00)
        x_11 = self.up_concatx11(x_20, x_10)
        x_21 = self.up_concatx21(x_30, x_20)
        x_31 = self.up_concatx31(x_40, x_30)
        # column : 2
        x_02 = self.up_concatx02(x_11, x_00, x_01)
        x_12 = self.up_concatx12(x_21, x_10, x_11)
        x_22 = self.up_concatx22(x_31, x_20, x_21)
        # column : 3
        x_03 = self.up_concatx03(x_12, x_00, x_01, x_02)
        x_13 = self.up_concatx13(x_22, x_10, x_11, x_12)
        # column : 4
        x_04 = self.up_concatx04(x_13, x_00, x_01, x_02, x_03)

        # column : 1
        y_01 = self.up_concaty01(y_10, y_00)
        y_11 = self.up_concaty11(y_20, y_10)
        y_21 = self.up_concaty21(y_30, y_20)
        y_31 = self.up_concaty31(y_40, y_30)
        # column : 2
        y_02 = self.up_concaty02(y_11, y_00, y_01)
        y_12 = self.up_concaty12(y_21, y_10, y_11)
        y_22 = self.up_concaty22(y_31, y_20, y_21)
        # column : 3
        y_03 = self.up_concaty03(y_12, y_00, y_01, y_02)
        y_13 = self.up_concaty13(y_22, y_10, y_11, y_12)
        # column : 4
        y_04 = self.up_concaty04(y_13, y_00, y_01, y_02, y_03)

        # final layer
        finalx_1 = self.finalx_1(x_01)
        finalx_2 = self.finalx_2(x_02)
        finalx_3 = self.finalx_3(x_03)
        finalx_4 = self.finalx_4(x_04)

        # final layer
        finaly_1 = self.finaly_1(y_01)
        finaly_2 = self.finaly_2(y_02)
        finaly_3 = self.finaly_3(y_03)
        finaly_4 = self.finaly_4(y_04)

        finalx = (finalx_1 + finalx_2 + finalx_3 + finalx_4) / 4
        finaly = (finaly_1 + finaly_2 + finaly_3 + finaly_4) / 4

        if self.is_ds:
            return finalx, finaly
        else:
            return finalx_4, finaly_4
