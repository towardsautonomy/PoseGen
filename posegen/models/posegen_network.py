from typing import Optional, List, Tuple

import torch
from torch import nn


from module import (
    DBlock,
    DBlockOptimized,
    GBlock,
    SNLinear,
)

from ..datatypes import CarTensorDataBatch


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

        self.block1 = DBlockOptimized(3, ndf >> 6)
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
        h1 = x
        h2 = self.block1(h1)
        h3 = self.block2(h2)
        h4 = self.block3(h3)
        h5 = self.block4(h4)
        h6 = self.block5(h5)
        h7 = self.block6(h6)
        h8 = self.block7(h7)
        h9 = self.activation(h8)
        h_final = torch.sum(h9, dim=(2, 3))
        y = self.l8(h_final)
        return y, [h1, h2, h3, h4, h5, h6, h7, h8, h9]


class Decoder(nn.Module):
    r"""
    ResNet backbone decoder architecture.
    Attributes:
        nz (int): latent vector dimension for upsampling.
        ngf (int): Variable controlling decoder feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(
        self, nz=128, ngf=512, bottom_width=4, skip_connections=False, n_encoders=2
    ):
        super().__init__()

        self.skip_connections = skip_connections

        self.l1 = nn.Linear(nz * n_encoders, (bottom_width ** 2) * ngf)
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

    def forward(self, x, enc_hidden_layers):
        (
            h1_enc,
            h2_enc,
            h3_enc,
            h4_enc,
            h5_enc,
            h6_enc,
            h7_enc,
            h8_enc,
            h9_enc,
        ) = enc_hidden_layers
        h1 = self.l1(x)
        h2 = self.unfatten(h1)
        if self.skip_connections:
            h2 += h7_enc
        h3 = self.block2(h2)
        if self.skip_connections:
            h3 += h6_enc
        h4 = self.block3(h3)
        if self.skip_connections:
            h4 += h5_enc
        h5 = self.block4(h4)
        if self.skip_connections:
            h5 += h4_enc
        h6 = self.block5(h5)
        if self.skip_connections:
            h6 += h3_enc
        h7 = self.block6(h6)
        if self.skip_connections:
            h7 += h2_enc
        h8 = self.block7(h7)
        h9 = self.b8(h8)
        h10 = self.activation(h9)
        h_final = self.c8(h10)
        y = torch.tanh(h_final)

        return y


class PoseGen(nn.Module):
    def __init__(
        self,
        ndf: int = 1024,
        ngf: int = 512,
        nz: int = 128,
        bottom_width: int = 4,
        skip_connections: bool = False,
        unconditional: bool = True,
        appearance_input: bool = False,
        bgnd_input: bool = False,
    ):
        super().__init__()
        self.unconditional = unconditional
        self.appearance_input = appearance_input
        self.bgnd_input = bgnd_input
        self.skip_connections = skip_connections
        self.nz = nz

        # object appearance encoder
        self.obj_appear_enc = Encoder(ndf, nz)
        # background encoder
        self.background_enc = Encoder(ndf, nz)
        # pose encoder
        self.pose_enc = Encoder(ndf, nz)
        # number of encoders
        self.n_encoders = 1 + int(self.appearance_input) + int(self.bgnd_input)
        self.dec = Decoder(
            nz,
            ngf,
            bottom_width,
            skip_connections=skip_connections,
            n_encoders=self.n_encoders,
        )

    @staticmethod
    def _sum_lists(*xs: Optional[List[torch.Tensor]]) -> List[torch.Tensor]:
        return [sum((h for h in hs if h is not None), []) for hs in zip(xs)]

    @staticmethod
    def _cat(*xs: Optional[torch.Tensor]) -> torch.Tensor:
        return torch.cat([x for x in xs if x is not None], dim=1)

    @staticmethod
    def _apply_encoder(
        cond: bool, enc: Encoder, data: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        return enc(data) if cond else None, None

    def forward(self, data: CarTensorDataBatch):
        if self.unconditional:
            z = torch.randn(len(data.car), self.nz, device=data.car.device)
            hidden_features = []
            self.skip_connections = False
            self.appearance_input = False
            self.bgnd_input = False
        else:
            z_pose, pose_hidden_features = self.pose_enc(data.pose)
            z_appear, appear_hidden_features = self._apply_encoder(
                self.appearance_input, self.obj_appear_enc, data.car
            )
            z_bgnd, bgnd_hidden_features = self._apply_encoder(
                self.bgnd_input, self.background_enc, data.background
            )
            hidden_features = self._sum_lists(
                pose_hidden_features, appear_hidden_features, bgnd_hidden_features
            )
            z = self._cat(z_pose, z_appear, z_bgnd)
        return self.dec(z, hidden_features)
