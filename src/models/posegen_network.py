import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *


class Encoder(nn.Module):
    r"""
    ResNet backbone encoder architecture.
    Attributes:
        ndf (int): Variable controlling encoder feature map sizes.
    """

    def __init__(self, ndf=1024, nz=128):
        super().__init__()

        self.block1 = DBlockOptimized(3, ndf >> 6)
        self.block2 = DBlock(ndf >> 6, ndf >> 5, downsample=True)
        self.block3 = DBlock(ndf >> 5, ndf >> 4, downsample=True)
        self.block4 = DBlock(ndf >> 4, ndf >> 3, downsample=True)
        self.block5 = DBlock(ndf >> 3, ndf >> 2, downsample=True)
        self.block6 = DBlock(ndf >> 2, ndf >> 1, downsample=True)
        self.block7 = DBlock(ndf >> 1, ndf, downsample=True)
        self.l8 = SNLinear(ndf, nz)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l8.weight.data, 1.0)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        y = self.l8(h)
        return y


class Decoder(nn.Module):
    r"""
    ResNet backbone decoder architecture.
    Attributes:
        nz (int): latent vector dimension for upsampling.
        ngf (int): Variable controlling decoder feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(self, nz=128, ngf=1024, bottom_width=4):
        super().__init__()

        self.l1 = nn.Linear(nz, (bottom_width ** 2) * ngf)
        self.unfatten = nn.Unflatten(1, (ngf, bottom_width, bottom_width))
        self.block2 = GBlock(ngf, ngf >> 1, upsample=True)
        self.block3 = GBlock(ngf >> 1, ngf >> 2, upsample=True)
        self.block4 = GBlock(ngf >> 2, ngf >> 3, upsample=True)
        self.block5 = GBlock(ngf >> 3, ngf >> 4, upsample=True)
        self.block6 = GBlock(ngf >> 4, ngf >> 5, upsample=True)
        self.block7 = GBlock(ngf >> 5, ngf >> 6, upsample=True)
        self.b8 = nn.BatchNorm2d(ngf >> 6)
        self.c8 = nn.Conv2d(ngf >> 6, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c8.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        h = self.unfatten(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.b8(h)
        h = self.activation(h)
        h = self.c8(h)
        y = torch.tanh(h)
        return y


class Generator(nn.Module):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(self, nz=128, ngf=1024, bottom_width=4):
        super().__init__()

        self.l1 = nn.Linear(nz, (bottom_width ** 2) * ngf)
        self.unfatten = nn.Unflatten(1, (ngf, bottom_width, bottom_width))
        self.block2 = GBlock(ngf, ngf >> 1, upsample=True)
        self.block3 = GBlock(ngf >> 1, ngf >> 2, upsample=True)
        self.block4 = GBlock(ngf >> 2, ngf >> 3, upsample=True)
        self.block5 = GBlock(ngf >> 3, ngf >> 4, upsample=True)
        self.block6 = GBlock(ngf >> 4, ngf >> 5, upsample=True)
        self.block7 = GBlock(ngf >> 5, ngf >> 6, upsample=True)
        self.b8 = nn.BatchNorm2d(ngf >> 6)
        self.c8 = nn.Conv2d(ngf >> 6, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c8.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        h = self.unfatten(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.b8(h)
        h = self.activation(h)
        h = self.c8(h)
        y = torch.tanh(h)
        return y


class Discriminator(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """

    def __init__(self, ndf=1024):
        super().__init__()

        self.block1 = DBlockOptimized(4, ndf >> 6)
        self.block2 = DBlock(ndf >> 6, ndf >> 5, downsample=True)
        self.block3 = DBlock(ndf >> 5, ndf >> 4, downsample=True)
        self.block4 = DBlock(ndf >> 4, ndf >> 3, downsample=True)
        self.block5 = DBlock(ndf >> 3, ndf >> 2, downsample=True)
        self.block6 = DBlock(ndf >> 2, ndf >> 1, downsample=True)
        self.block7 = DBlock(ndf >> 1, ndf, downsample=True)
        self.l8 = SNLinear(ndf, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l8.weight.data, 1.0)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        y = self.l8(h)
        return y


class AutoEncoder(nn.Module):
    def __init__(self, ndf=1024, ngf: int = 1024, nz: int=128, bottom_width: int = 4):
        super().__init__()
        self.enc = Encoder(ndf, nz)
        self.dec = Decoder(nz, ngf, bottom_width)

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        return out
