from typing import Optional, List, Tuple

import torch
from torch import nn


from module import (
    DBlock,
    DBlockOptimized,
    GBlock,
    SNLinear,
)

from ..datatypes import ObjectTensorDataBatch


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

    def __init__(self, ndf: int):
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
        ndf: int,
        ngf: int,
        nz: int,
        bottom_width: int,
        skip_connections: bool,
        condition_on_object: bool,
        condition_on_pose: bool,
        condition_on_background: bool,
        # use_random_z_if_no_input: bool = False,  # TODO: add this
    ):
        super().__init__()
        self.condition_on_object = condition_on_object
        self.condition_on_pose = condition_on_pose
        self.condition_on_background = condition_on_background
        self.skip_connections = skip_connections
        self.nz = nz

        self.enc_object = Encoder(ndf, nz)
        self.enc_pose = Encoder(ndf, nz)
        self.enc_background = Encoder(ndf, nz)

        # number of encoders
        self.n_encoders = (
            int(self.condition_on_object)
            + int(self.condition_on_pose)
            + int(self.condition_on_background)
        )
        self.dec = Decoder(
            nz,
            ngf,
            bottom_width,
            skip_connections=skip_connections,
            n_encoders=max(self.n_encoders, 1),
        )

    @staticmethod
    def _sum_lists(*xs: Optional[List[torch.Tensor]]) -> List[torch.Tensor]:
        xs_nones_removed = (x for x in xs if x is not None)
        return [sum(hs) for hs in zip(*xs_nones_removed)]

    @staticmethod
    def _cat(*xs: Optional[torch.Tensor]) -> torch.Tensor:
        return torch.cat([x for x in xs if x is not None], dim=1)

    @staticmethod
    def _apply_encoder(
        cond: bool, enc: Encoder, data: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        return enc(data) if cond else None, None

    def forward(self, data: ObjectTensorDataBatch):
        if self.n_encoders == 0:
            z = torch.randn(len(data.object), self.nz, device=data.object.device)
            h = []
        else:
            z_o, h_o = self._apply_encoder(
                self.condition_on_object, self.enc_object, data.object
            )
            z_p, h_p = self._apply_encoder(
                self.condition_on_pose, self.enc_pose, data.pose
            )
            z_b, h_b = self._apply_encoder(
                self.condition_on_background, self.enc_background, data.background
            )
            h = self._sum_lists(h_o, h_p, h_b)
            z = self._cat(z_o, z_p, z_b)
        return self.dec(z, h)
