from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ConfigCommon:
    ndf: int
    ngf: int
    bottom_width: int
    out_path: str
    lr: float
    betas: Tuple[float, float]
    nz: int
    batch_size: int
    num_workers: int
    repeat_d: int
    max_steps: int
    seed: int
    eval_every: int
    ckpt_every: int


@dataclass(frozen=True)
class Config:
    dataset: str
    lambda_gan: float
    lambda_full: float
    lambda_object: float
    lambda_background: float
    pretrain: bool
    condition_on_object: bool
    condition_on_pose: bool
    condition_on_background: bool
    skip_connections: bool
