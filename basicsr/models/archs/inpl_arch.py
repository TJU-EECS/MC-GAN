import torch
import numpy as np
from torch.autograd import Variable
from torch import nn as nn
from torch.nn import functional as F


class MyConv2D(nn.Module):
    def __init__(self, inc, nc, kernel, stride, dilation=1, bias=True, act=None, norm=None, mode='zeros'):
        super(MyConv2D, self).__init__()
        pad_num = (dilation*(kernel-1) + 2 - stride)//2
        if act == 'relu':
            self.act = nn.ReLU(inplace=False)
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        elif act == 'prelu':
            self.act = nn.PReLU(num_parameters=nc, init=0.1)
        elif act == 'elu':
            self.act = nn.ELU(inplace=False)

        if norm == 'batchnorm':
            self.norm = nn.BatchNorm2d(num_features=nc)
        elif norm == 'instancenorm':
            self.norm = nn.InstanceNorm2d(num_features=nc, affine=True, track_running_stats=True)

        self.conv = nn.Conv2d(
            in_channels=inc,
            out_channels=nc,
            kernel_size=kernel,
            stride=stride,
            padding=pad_num,
            padding_mode=mode,
            dilation=dilation,
            bias=bias
        )

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        if hasattr(self, 'act'):
            x = self.act(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, nc, z_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            MyConv2D(nc, nc, kernel=3, stride=1, dilation=1, act='lrelu'),
            MyConv2D(nc, nc * 2, kernel=4, stride=2, dilation=1),
            MyConv2D(nc * 2, nc * 2, kernel=3, stride=1, dilation=1, act='lrelu'),
            MyConv2D(nc * 2, nc * 4, kernel=4, stride=2, dilation=1),
            MyConv2D(nc * 4, nc * 4, kernel=3, stride=1, dilation=1, act='lrelu'),
            MyConv2D(nc * 4, nc * 8, kernel=4, stride=2, dilation=1),
            MyConv2D(nc * 8, nc * 8, kernel=3, stride=1, dilation=1, act='lrelu'),
            MyConv2D(nc * 8, nc * 8, kernel=4, stride=2, dilation=1),
            MyConv2D(nc * 8, nc * 8, kernel=3, stride=1, dilation=1)
        )

        self.VAE_encoder = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(1024, z_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x_vector = self.VAE_encoder(x.view(x.size()[0], -1))
        return x_vector


class LatentCodeFusion(nn.Module):
    def __init__(self, nc, z_dim):
        super(LatentCodeFusion, self).__init__()
        self.latent_0 = AutoEncoder(nc, z_dim)
        self.latent_1 = AutoEncoder(nc, z_dim)
        self.latent_2 = AutoEncoder(nc, z_dim)
        self.latent_3 = AutoEncoder(nc, z_dim)

        self.latent_fusion = nn.Sequential(
            nn.Linear(z_dim * 4, z_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(z_dim, 1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(1024, 4096),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(4096, 8192)
        )

        self.decoder = nn.Sequential(
            MyConv2D(nc * 8, nc * 8, kernel=3, stride=1, dilation=1, act='lrelu'),
            nn.ConvTranspose2d(nc * 8, nc * 4, kernel_size=4, stride=2, padding=1, bias=True),
            MyConv2D(nc * 4, nc * 4, kernel=3, stride=1, dilation=1, act='lrelu'),
            nn.ConvTranspose2d(nc * 4, nc * 2, kernel_size=4, stride=2, padding=1, bias=True),
            MyConv2D(nc * 2, nc * 2, kernel=3, stride=1, dilation=1, act='lrelu'),
            nn.ConvTranspose2d(nc * 2, nc, kernel_size=4, stride=2, padding=1, bias=True),
            MyConv2D(nc, nc, kernel=3, stride=1, dilation=1, act='lrelu'),
            nn.ConvTranspose2d(nc, nc, kernel_size=4, stride=2, padding=1, bias=True),
            MyConv2D(nc, nc, kernel=3, stride=1, dilation=1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        z_0, z_1, z_2, z_3 = self.latent_0(x), self.latent_1(x), self.latent_2(x), self.latent_3(x)
        z_fusion = self.latent_fusion(torch.cat((z_0, z_1, z_2, z_3), dim=1))
        res = self.decoder(z_fusion.view(b, c * 8, 4, 4))
        return x - res


class BodyBlock(nn.Module):
    def __init__(self, nc, scale):
        super(BodyBlock, self).__init__()
        self.filed_7x7 = MyConv2D(nc, nc, kernel=7, stride=1, dilation=1, act='lrelu')
        self.filed_5x5 = MyConv2D(nc, nc, kernel=5, stride=1, dilation=1, act='lrelu')
        self.filed_3x3 = MyConv2D(nc, nc, kernel=3, stride=1, dilation=1, act='lrelu')

        self.dilation_3 = MyConv2D(nc, nc, kernel=3, stride=1, dilation=3, act='lrelu')
        self.dilation_2 = MyConv2D(nc, nc, kernel=3, stride=1, dilation=2, act='lrelu')
        self.dilation_1 = MyConv2D(nc, nc, kernel=3, stride=1, dilation=1, act='lrelu')

        self.filed_fusion = MyConv2D(nc * 3, nc, kernel=3, stride=1, dilation=1, act='lrelu')
        self.dilation_fusion = MyConv2D(nc * 3, nc, kernel=3, stride=1, dilation=1, act='lrelu')

        self.fusion = MyConv2D(nc * 2, nc, kernel=3, stride=1, dilation=1)
        self.scale = scale

    def forward(self, x):
        filed_7, filed_5, filed_3 = self.filed_7x7(x), self.filed_5x5(x), self.filed_3x3(x)
        dilat_3, dilat_2, dilat_1 = self.dilation_3(x), self.dilation_2(x), self.dilation_1(x)
        filed_fusion = self.filed_fusion(torch.cat((filed_7, filed_5, filed_3), dim=1))
        dilat_fusion = self.dilation_fusion(torch.cat((dilat_3, dilat_2, dilat_1), dim=1))
        fusion = self.fusion(torch.cat((filed_fusion, dilat_fusion), dim=1))
        return x + fusion


class ResiduleBlock(nn.Module):
    def __init__(self, nc, scale):
        super(ResiduleBlock, self).__init__()
        self.conv_2 = MyConv2D(nc, nc, kernel=3, stride=1, dilation=1, act='lrelu')
        self.conv_1 = MyConv2D(nc, nc, kernel=3, stride=1, dilation=1)

    def forward(self, x):
        res = self.conv_1(self.conv_2(x))
        return x + res


class INPL(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Currently, it supports x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 32
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=32):
        super(INPL, self).__init__()
        self.head = MyConv2D(num_in_ch, num_feat, kernel=3, stride=1, dilation=1)
        self.de_noise = LatentCodeFusion(num_feat, 512)
        body = []
        for _ in range(num_block):
            body.append(BodyBlock(num_feat, scale=0.1))
            #body.append(ResiduleBlock(num_feat, scale=0.1))
        self.body = nn.Sequential(*body)
        self.tail = MyConv2D(num_feat, num_out_ch, kernel=3, stride=1, dilation=1)

    def forward(self, x):
        x = self.head(x)
        x = self.de_noise(x)
        x = self.body(x)
        x = self.tail(x)
        return x, None, None
