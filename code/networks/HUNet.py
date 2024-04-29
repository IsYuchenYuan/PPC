

# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import UnetConv3, UnetUp3, UnetUp3_CT, unetConv2, unetUp, init_weights
from networks.unetmodel import UNet_DS
from einops import rearrange

def converToSlice(input):

    D = input.size(-1)
    input2d = input[..., 0]
    for i in range(1, D):
        input2dtmp = input[..., i]
        input2d = torch.cat((input2d, input2dtmp), dim=0)

    return input2d

class unet_2D(nn.Module):
    def __init__(self, feature_scale=4, n_classes=2, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_2D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, ks=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, ks=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, ks=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, ks=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm, ks=3, padding=1)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        self.sigmoid = nn.Sigmoid()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.dropout1(center)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        final = self.final(up1)
        if self.n_classes == 1:
            final = self.sigmoid(final)
        return up1, final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class unet_3D(nn.Module):

    def __init__(self, feature_scale=1, n_classes=2, is_deconv=True, in_channels=3, is_batchnorm=True, use_multiscale=True):
        super(unet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        self.use_multi = use_multiscale

        filters = [16, 32, 64, 128, 256]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)
        # final conv (without any concat)
        if self.use_multi:
            self.final = UnetConv3(64, 64, self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        else:
            self.final = UnetConv3(16, 16, self.is_batchnorm, kernel_size=(
                3, 3, 3), padding_size=(1, 1, 1))

        # self.cls = nn.Sequential(
        #     nn.Conv3d(16, 16, (3, 3, 3), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(16, self.n_classes, (1, 1, 1))
        # )

        self.upsample4 = nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear')
        self.upsample3 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.fusion = UnetConv3(240, 64, self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        self.sigmoid = nn.Sigmoid()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.dropout1(center)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        if self.use_multi:
            fusion = self.fusion(
                torch.cat((self.upsample4(up4),
                           self.upsample3(up3),
                           self.upsample2(up2),
                           up1),dim=1)
            )

            final = self.final(fusion)

        else:
            final = self.final(up1)

        # final = self.cls(final)
        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class UnetUp3_CT_smallz(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT_smallz, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

class unet_3D_smallz(nn.Module):

    def __init__(self, feature_scale=1, n_classes=2, is_deconv=True, in_channels=3, is_batchnorm=True, use_multiscale=True):
        super(unet_3D_smallz, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        self.use_multi = use_multiscale

        filters = [16, 32, 64, 128, 256]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT_smallz(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT_smallz(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)
        # final conv (without any concat)
        if self.use_multi:
            self.final = UnetConv3(64, 64, self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        else:
            self.final = UnetConv3(16, 64, self.is_batchnorm, kernel_size=(
                3, 3, 3), padding_size=(1, 1, 1))

        self.upsample4 = nn.Upsample(scale_factor=(8, 8, 4), mode='trilinear')
        self.upsample3 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.fusion = UnetConv3(240, 64, self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        self.sigmoid = nn.Sigmoid()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.dropout1(center)
        up4 = self.up_concat4(conv4, center)
        # print(up4.shape)
        up3 = self.up_concat3(conv3, up4)
        # print(up3.shape)
        up2 = self.up_concat2(conv2, up3)
        # print(up2.shape)
        up1 = self.up_concat1(conv1, up2)
        # print(up1.shape)
        up1 = self.dropout2(up1)

        if self.use_multi:
            fusion = self.fusion(
                torch.cat((self.upsample4(up4),
                           self.upsample3(up3),
                           self.upsample2(up2),
                           up1),dim=1)
            )

            final = self.final(fusion)
        else:
            final = self.final(up1)
        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p

def select(x, index):
    """
    BxD,C,H,W->B,C,D,H,W
    """
    tmp = x[index, ...]
    tmp = tmp.unsqueeze(dim=1)
    return tmp


class dense_rnn_net(nn.Module):
    def __init__(self, inchannel, nclasses):
        super(dense_rnn_net, self).__init__()
        self.inchannel = inchannel
        self.nclasses = nclasses
        self.unet2d = UNet_DS(self.inchannel, self.nclasses)
        self.unet3d = unet_3D(in_channels=1,n_classes=self.nclasses)

        self.final = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, self.nclasses, (1, 1, 1))
        )

    def forward(self, x_3d):

        B, C, H, W, D = x_3d.size()
        return_dict = {}
        x_2d = converToSlice(x_3d)
        feature2d, classifer2d = self.unet2d(x_2d)  # feat(b*d,64,h,w) cls(b*d,2,h,w)
        return_dict["cls_2d"] = classifer2d
        feature2To3 = rearrange(
            feature2d, "(b d) c h w -> b c h w d", b=B
        )

        # input3d = torch.cat((x_3d, classifer2To3), dim=1)
        feature3d = self.unet3d(x_3d)
        hybridfeature = torch.add(feature3d, feature2To3)
        embedding = rearrange(hybridfeature, "b c h w d -> (b d) c h w ")
        # return high level semantic features to refine pseudo labels
        return_dict["feature"] = embedding
        finalout = self.final(hybridfeature)
        return_dict["cls_3d"] = finalout
        return return_dict
        # return output3d


class unet3D_prob(nn.Module):

    def __init__(self, feature_scale=1, n_classes=2, outdim=128, is_deconv=True, in_channels=3, is_batchnorm=True, use_multiscale=False):
        super(unet3D_prob, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        self.use_multi = use_multiscale

        filters = [16, 32, 64, 128, 256]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)
        # final conv (without any concat)
        if self.use_multi:
            self.representation = nn.Sequential(
                                    nn.Conv3d(64, 64, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(64, outdim, (1, 1, 1))
                                )
        else:
            self.representation = nn.Sequential(
                                    nn.Conv3d(16, 64, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(64, outdim, (1, 1, 1))
                                )


        self.cls =  nn.Conv3d(16, n_classes, (3, 3, 3), padding=(1, 1, 1))

        self.upsample4 = nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear')
        self.upsample3 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.fusion = UnetConv3(240, 64, self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        self.sigmoid = nn.Sigmoid()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.dropout1(center)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        if self.use_multi:
            fusion = self.fusion(
                torch.cat((self.upsample4(up4),
                           self.upsample3(up3),
                           self.upsample2(up2),
                           up1),dim=1)
            )

            representation = self.representation(fusion)
            cls_res = self.cls(fusion)
        else:
            representation = self.representation(up1)
            cls_res = self.cls(up1)


        return cls_res,representation,up1


if __name__ == '__main__':
    a = torch.randn(1, 1, 64, 64, 32)
    net = unet_3D()
    b = net(a)
    # print(b.size())
    # a = torch.randn(1,3,64,64)
    # net = dense_rnn_net()
    # b=net(a)