from . import ConfigCommon


config = ConfigCommon(
    ndf=1024,
    ngf=512,
    bottom_width=4,
    out_path="out",
    lr=2e-4,
    betas=(0.0, 0.9),
    nz=256,
    batch_size=64,
    num_workers=8,
    repeat_d=1,
    max_steps=75_000,
    seed=0,
    eval_every=500,
    ckpt_every=500,
)
