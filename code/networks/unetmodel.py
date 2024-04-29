# -*- coding: utf-8 -*-
"""
An implementation of the U-Net paper:
    Olaf Ronneberger, Philipp Fischer, Thomas Brox:
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    MICCAI (3) 2015: 234-241
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
"""
from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=False):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        # self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
        #                           kernel_size=3, padding=1)
        # self.representation = nn.Sequential(
        #     nn.Conv2d(self.ft_chns[0], self.ft_chns[0], 3, padding=1, bias=False),
        #     nn.BatchNorm2d(self.ft_chns[0]),
        #     nn.ReLU(),
        #     nn.Conv2d(self.ft_chns[0], 64, 1)
        # )

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_up1 = self.up1(x4, x3)
        x_up2 = self.up2(x_up1, x2)
        x_up3 = self.up3(x_up2, x1)
        x_up4 = self.up4(x_up3, x0)
        # output = self.out_conv(x_up4)
        return  x_up1,x_up2,x_up3,x_up4



class Decoder_DS(nn.Module):
    def __init__(self, params):
        super(Decoder_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)


        self.dropout = nn.Dropout2d(0.5)

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x4 = self.dropout(x4)
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        x = self.dropout(x)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class UNet_DS(nn.Module):
    def __init__(self, in_chns, class_num, use_multi=True):
        super().__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.use_multi = use_multi

        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels=240,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        if self.use_multi:
            self.finallayer = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        else:
            self.finallayer = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.cls = nn.Conv2d(64, class_num, kernel_size=3, padding=1)

        # self.representation = nn.Sequential(
        #     nn.Conv2d(64, 64, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, out_dim, 1)
        # )


    def forward(self, x):
        feature = self.encoder(x)
        x_up1, x_up2, x_up3, x_up4 = self.decoder(feature)
        if self.use_multi:
            x_fusion =self.fuse(
                torch.cat((self.up1(x_up1),self.up2(x_up2),self.up3(x_up3),x_up4),dim=1))
            feat = self.finallayer(x_fusion)
        else:
            feat = self.finallayer(x_up4)
        cls_res = self.cls(feat)
        # representation  = self.representation(feat)
        return feat,cls_res


class UNet(nn.Module):
    def __init__(self, in_chns, class_num, out_dim):
        super().__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.cls = nn.Conv2d(16, class_num, kernel_size=3, padding=1)

        self.representation = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_dim, 1)
        )


    def forward(self, x):
        feature = self.encoder(x)
        x_up1, x_up2, x_up3, x_up4 = self.decoder(feature)

        cls_res = self.cls(x_up4)
        representation  = self.representation(x_up4)
        return cls_res, representation, x_up4

# output the mu and sigma from the same encoder
class UNet_unify(nn.Module):
    def __init__(self, in_chns, class_num, embed_dim=256, sigma_mode="radius", sigma_trans_mode='sigmoid_learn'):
        super().__init__()

        self.embed_dim = embed_dim
        self.sigma_mode = sigma_mode
        self.sigma_transform_mode = sigma_trans_mode

        if not sigma_mode in ["constant", "radius", "diagonal", "full"]:
            print("The sigma mode " + str(sigma_mode) + " is not supported.")
            assert False

        sigma_dim_dictionary = {
            "constant": 0,
            "radius": 1,
            "diagonal": embed_dim,
            "full": int(embed_dim * (embed_dim + 1) / 2)
        }

        sigma_dim = sigma_dim_dictionary[sigma_mode]
        self.sigma_dim = sigma_dim

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.cls = nn.Conv2d(16, class_num, kernel_size=3, padding=1)

        self.representation = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim + sigma_dim, 1)
        )
        if self.sigma_transform_mode == 'softplus_learn':
            self.scale = Parameter(torch.Tensor([1.0]))
            self.offset = Parameter(torch.Tensor([1.0]))
            self.div = Parameter(torch.Tensor([1.0]))
        elif self.sigma_transform_mode == 'sigmoid_learn':
            self.scale = Parameter(torch.Tensor([1.0]))
            self.offset = Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        feature = self.encoder(x)
        x_up1, x_up2, x_up3, x_up4 = self.decoder(feature)

        cls_res = self.cls(x_up4)
        X_encoded  = self.representation(x_up4)

        if self.sigma_dim > 0:
            mu, sigma_raw = torch.split(X_encoded, [self.embed_dim, self.sigma_dim], 1)
            # v7 - basic stuff in a narrow range with softplus
            if self.sigma_transform_mode == "softplus_narrow":
                offset = 1.0
                scale = 1.0
                sigma = offset + scale * F.softplus(sigma_raw)

            # v8 - basic stuff in a narrow range with sigmoid
            elif self.sigma_transform_mode == "sigmoid_narrow":
                offset = 1.0
                scale = 1.0
                sigma = offset + scale * F.sigmoid(sigma_raw)

            # v9 -- basic, wider range
            elif self.sigma_transform_mode == "sigmoid_wide":
                offset = 1.0
                scale = 4.0
                sigma = offset + scale * F.sigmoid(sigma_raw)

            # v10 -- learnable
            elif self.sigma_transform_mode == "softplus_learn":
                sigma = self.offset + self.scale * F.softplus(sigma_raw / self.div)

            # v10 -- learnable
            elif self.sigma_transform_mode == "sigmoid_learn":
                sigma = self.offset + self.scale * F.sigmoid(sigma_raw)

        else:
            mu = X_encoded

        if self.sigma_mode == "full":  # NOT FULL IMPLEMENTED AS NOT USEFUL
            assert False

        elif self.sigma_mode in ["radius", "diagonal"]:
            sigma = sigma

        elif self.sigma_mode == "constant":
            sigma = torch.ones_like(X_encoded)

        return cls_res, mu, sigma # the the inverse of sigma_sq