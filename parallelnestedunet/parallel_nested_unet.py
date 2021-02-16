import torch
import torch.nn as nn
from torch.nn import init

from bridge import Bridge


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
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
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
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


class ParallelNestedUNet(nn.Module):

    def __init__(self, n_channels=1, n_classes=2, feature_scale=1, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(ParallelNestedUNet, self).__init__()
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

        self.bridge0 = Bridge(filters[0])
        self.bridge1 = Bridge(filters[1])
        self.bridge2 = Bridge(filters[2])
        self.bridge3 = Bridge(filters[3])

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
        X_00 = self.convx00(inputs)  # 16*512*512
        Y_00 = self.convy00(inputs)
        maxpoolx0 = self.maxpool(X_00)  # 16*256*256
        maxpooly0 = self.maxpool(Y_00)
        maxpool0 = self.bridge0(maxpoolx0, maxpooly0)

        X_10 = self.convx10(maxpool0)  # 32*256*256
        Y_10 = self.convy10(maxpool0)
        maxpoolx1 = self.maxpool(X_10)  # 32*128*128
        maxpooly1 = self.maxpool(Y_10)
        maxpool1 = self.bridge1(maxpoolx1, maxpooly1)

        X_20 = self.convx20(maxpool1)  # 64*128*128
        Y_20 = self.convy20(maxpool1)
        maxpoolx2 = self.maxpool(X_20)  # 64*64*64
        maxpooly2 = self.maxpool(Y_20)
        maxpool2 = self.bridge2(maxpoolx2, maxpooly2)

        X_30 = self.convx30(maxpool2)  # 128*64*64
        Y_30 = self.convy30(maxpool2)
        maxpoolx3 = self.maxpool(X_30)  # 128*32*32
        maxpooly3 = self.maxpool(Y_30)
        maxpool3 = self.bridge3(maxpoolx3, maxpooly3)

        X_40 = self.convx40(maxpool3)  # 256*32*32
        Y_40 = self.convy40(maxpool3)

        # column : 1
        X_01 = self.up_concatx01(X_10, X_00)
        X_11 = self.up_concatx11(X_20, X_10)
        X_21 = self.up_concatx21(X_30, X_20)
        X_31 = self.up_concatx31(X_40, X_30)
        # column : 2
        X_02 = self.up_concatx02(X_11, X_00, X_01)
        X_12 = self.up_concatx12(X_21, X_10, X_11)
        X_22 = self.up_concatx22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concatx03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concatx13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concatx04(X_13, X_00, X_01, X_02, X_03)

        # column : 1
        Y_01 = self.up_concaty01(Y_10, Y_00)
        Y_11 = self.up_concaty11(Y_20, Y_10)
        Y_21 = self.up_concaty21(Y_30, Y_20)
        Y_31 = self.up_concaty31(Y_40, Y_30)
        # column : 2
        Y_02 = self.up_concaty02(Y_11, Y_00, Y_01)
        Y_12 = self.up_concaty12(Y_21, Y_10, Y_11)
        Y_22 = self.up_concaty22(Y_31, Y_20, Y_21)
        # column : 3
        Y_03 = self.up_concaty03(Y_12, Y_00, Y_01, Y_02)
        Y_13 = self.up_concaty13(Y_22, Y_10, Y_11, Y_12)
        # column : 4
        Y_04 = self.up_concaty04(Y_13, Y_00, Y_01, Y_02, Y_03)

        # final layer
        finalx_1 = self.finalx_1(X_01)
        finalx_2 = self.finalx_2(X_02)
        finalx_3 = self.finalx_3(X_03)
        finalx_4 = self.finalx_4(X_04)

        # final layer
        finaly_1 = self.finaly_1(Y_01)
        finaly_2 = self.finaly_2(Y_02)
        finaly_3 = self.finaly_3(Y_03)
        finaly_4 = self.finaly_4(Y_04)

        finalx = (finalx_1 + finalx_2 + finalx_3 + finalx_4) / 4
        finaly = (finaly_1 + finaly_2 + finaly_3 + finaly_4) / 4

        if self.is_ds:
            return finalx, finaly
        else:
            return finalx_4, finaly_4
