from dataclasses import dataclass
import functools
import os
from pathlib import Path
from typing import Callable, Dict, Tuple, Union

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tbx
from torch import optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import wandb

from . import datasets
from .datatypes import Lambdas, ObjectTensorDataBatch, Split
from .datasets import ObjectDataset
from .experiments import Config, ConfigCommon
from .metrics import Metrics, MetricCalculator
from .models import Discriminator, PoseGen
from .utils import binarize_pose, get_device, get_logger


logger = get_logger(__name__)
LossGeneratorFn = Callable[[torch.Tensor], torch.Tensor]
LossDiscriminatorFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def compute_prob(logits: torch.Tensor) -> torch.Tensor:
    r"""
    Computes probability from model output.
    """

    return torch.sigmoid(logits).mean()


def hinge_loss_g(fake_preds: torch.Tensor) -> torch.Tensor:
    r"""
    Computes generator hinge loss.
    """

    return -fake_preds.mean()


def hinge_loss_d(real_preds: torch.Tensor, fake_preds: torch.Tensor) -> torch.Tensor:
    r"""
    Computes discriminator hinge loss.
    """

    return F.relu(1.0 - real_preds).mean() + F.relu(1.0 + fake_preds).mean()


def compute_loss_g(
    net_g: PoseGen,
    net_d: Discriminator,
    real: ObjectTensorDataBatch,
    loss_func_g: LossGeneratorFn,
    lambdas: Lambdas,
    pretrain: bool,
) -> Tuple[torch.Tensor, ...]:
    """
    General implementation to compute generator loss.
    TODO: add L_background
    """
    fakes = net_g(real)
    fake_preds = net_d(fakes).view(-1)
    loss_g = lambdas.gan * loss_func_g(fake_preds)

    # different flavors of reconstruction losses
    # these losses are additive and independent
    # note that lambda_mse needs to be scaled outside of this function
    # if more than one kind of MSE loss applies
    mse = torch.nn.MSELoss(reduction="mean")
    if pretrain:
        # full images reconstruction loss
        loss_g += lambdas.full * mse(real.object, fakes)

    if net_g.condition_on_pose:
        # masked object reconstruction loss
        pose_binary = binarize_pose(real.pose)
        if lambdas.obj > 0:
            masked_fakes = fakes * pose_binary
            masked_real = real.object * pose_binary
            loss_g += lambdas.obj * mse(masked_fakes, masked_real)

    return loss_g, fakes, fake_preds


def compute_loss_d(
    net_g: PoseGen,
    net_d: Discriminator,
    real: ObjectTensorDataBatch,
    loss_func_d: LossDiscriminatorFn,
) -> Tuple[torch.Tensor, ...]:
    r"""
    General implementation to compute discriminator loss.
    """

    real_preds = net_d(real.object).view(-1)
    fakes = net_g(real).detach()
    fake_preds = net_d(fakes).view(-1)
    loss_d = loss_func_d(real_preds, fake_preds)
    return loss_d, fakes, real_preds, fake_preds


def train_step(
    net: Union[PoseGen, Discriminator], opt: Optimizer, sch: LambdaLR, compute_loss
):
    r"""
    General implementation to perform a training step.
    """

    net.train()
    loss = compute_loss()
    net.zero_grad()
    loss.backward()
    opt.step()
    sch.step()
    return loss


def evaluate(
    net_g: PoseGen,
    net_d: Discriminator,
    ds: ObjectDataset,
    dataloader: DataLoader,
    device: torch.device,
    lambdas: Lambdas,
    pretrain: bool,
) -> Tuple[Metrics, ObjectTensorDataBatch]:
    """
    Evaluates model and logs metrics.
    """

    net_g.to(device).eval()
    net_d.to(device).eval()

    with torch.no_grad():
        # Initialize metrics
        metric_calculator = MetricCalculator(device, ds.transform_reverse_fn_objects)

        for real in tqdm(dataloader, desc="Evaluating Model"):
            real = real.to(device)

            loss_d, fakes, real_pred, fake_pred = compute_loss_d(
                net_g,
                net_d,
                real,
                hinge_loss_d,
            )
            loss_g, _, _ = compute_loss_g(
                net_g, net_d, real, hinge_loss_g, lambdas, pretrain
            )
            metric_calculator.update(
                real=real,
                fake_object=fakes,
                loss_g=loss_g,
                loss_d=loss_d,
                preds_real=real_pred,
                preds_fake=fake_pred,
            )

        # Process metrics
        metrics = metric_calculator.compute()
    return metrics, real


def get_samples(
    net_g: PoseGen, real: ObjectTensorDataBatch
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    true_samples = real.object.cpu()
    true_samples = vutils.make_grid(true_samples, nrow=8, padding=4, normalize=True)
    # bgnd_samples = real_bgnd.cpu()
    # bgnd_samples = vutils.make_grid(bgnd_samples, nrow=8, padding=4, normalize=True)
    pose_samples = real.pose.cpu()
    pose_samples = vutils.make_grid(pose_samples, nrow=8, padding=4, normalize=True)
    fake_samples = net_g(real)
    fake_samples = F.interpolate(fake_samples, 256).cpu()
    fake_samples = vutils.make_grid(fake_samples, nrow=8, padding=4, normalize=True)
    return true_samples, pose_samples, fake_samples


@dataclass
class Trainer:
    dataset: str
    lambda_gan: float
    lambda_full: float
    lambda_object: float
    lambda_background: float
    pretrain: bool
    condition_on_object: bool
    condition_on_pose: bool
    condition_on_background: bool
    seed: int
    skip_connections: bool
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
    eval_every: int
    ckpt_every: int
    resume: bool
    name: str
    device: torch.device

    @classmethod
    def from_configs(
        cls, params_common: ConfigCommon, params: Config, name: str
    ) -> "Trainer":
        device = get_device()
        rest = dict(device=device, name=name, resume=False)
        args = {**params_common.__dict__, **params.__dict__, **rest}
        return cls(**args)

    @property
    def dataset_loader_fns(self) -> Dict[str, Callable[[Split], ObjectDataset]]:
        return dict(
            stanford_cars=datasets.get_stanford_cars_dataset,
            tesla=datasets.get_tesla_dataset,
        )

    @property
    def exp_dir(self) -> Path:
        return Path(self.out_path) / Path(self.name)

    @property
    def log_dir(self) -> Path:
        return self.exp_dir / Path("logs")

    @property
    def ckpt_dir(self) -> Path:
        return self.exp_dir / Path("ckpt")

    def __post_init__(self):
        self._dir_checks()
        self.lambdas = Lambdas(
            gan=self.lambda_gan, full=self.lambda_full, obj=self.lambda_object
        )

        # dataset
        ds_loader = self.dataset_loader_fns[self.dataset]
        self.ds_train = ds_loader(Split.train)
        args_dl = dict(
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        self.train_dl = self.ds_train.get_dataloader(**args_dl)
        self.validation_dl = ds_loader(Split.validation).get_dataloader(**args_dl)
        self.test_dl = ds_loader(Split.test).get_dataloader(**args_dl)

        torch.manual_seed(self.seed)
        self.net_g = self._instantiate_net_g().to(self.device)
        self.net_d = Discriminator(ndf=self.ndf).to(self.device)

        self.opt_g = optim.Adam(self.net_g.parameters(), self.lr, self.betas)
        self.opt_d = optim.Adam(self.net_d.parameters(), self.lr, self.betas)

        self.sch_g = optim.lr_scheduler.LambdaLR(
            self.opt_g, lr_lambda=lambda s: 1.0 - ((s * self.repeat_d) / self.max_steps)
        )
        self.sch_d = optim.lr_scheduler.LambdaLR(
            self.opt_d, lr_lambda=lambda s: 1.0 - (s / self.max_steps)
        )

        self.step = 0
        self.logger = tbx.SummaryWriter(str(self.log_dir))

    def _instantiate_net_g(self) -> PoseGen:
        return PoseGen(
            ndf=self.ndf,
            ngf=self.ngf,
            bottom_width=self.bottom_width,
            skip_connections=self.skip_connections,
            condition_on_object=self.condition_on_object,
            condition_on_pose=self.condition_on_pose,
            condition_on_background=self.condition_on_background,
            nz=self.nz,
        )

    def _dir_checks(self) -> None:
        if not self.resume and self.exp_dir.exists():
            raise FileExistsError(
                f"experiment {self.name} already exists. either resume or pass a new experiment name."
            )
        for p in (
            self.out_path,
            self.exp_dir,
            self.log_dir,
            self.ckpt_dir,
        ):
            Path(p).mkdir(exist_ok=True)

    def _state_dict(self):
        return {
            "net_g": self.net_g.state_dict(),
            "net_d": self.net_d.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "sch_g": self.sch_g.state_dict(),
            "sch_d": self.sch_d.state_dict(),
            "step": self.step,
        }

    def _load_state_dict(self, state_dict):
        self.net_g.load_state_dict(state_dict["net_g"])
        self.net_d.load_state_dict(state_dict["net_d"])
        self.opt_g.load_state_dict(state_dict["opt_g"])
        self.opt_d.load_state_dict(state_dict["opt_d"])
        self.sch_g.load_state_dict(state_dict["sch_g"])
        self.sch_d.load_state_dict(state_dict["sch_d"])
        self.step = state_dict["step"]

    def _load_checkpoint(self):
        """
        Finds the last checkpoint in ckpt_dir and load states.
        """

        ckpt_paths = [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pth")]
        if ckpt_paths:  # Train from scratch if no checkpoints were found
            ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_path)
            self._load_state_dict(torch.load(ckpt_path))

    def _save_checkpoint(self):
        """
        Saves model, optimizer and trainer states.
        """
        ckpt_path = os.path.join(self.ckpt_dir, f"{self.step}.pth")
        torch.save(self._state_dict(), ckpt_path)

    def _log(self, metrics: Metrics, real: ObjectTensorDataBatch) -> None:
        r"""
        Logs metrics and samples to Tensorboard.
        """

        for k, v in metrics.get_dict("validation").items():
            self.logger.add_scalar(k, v, self.step)
        obj, pose, fake = get_samples(self.net_g, real)
        self.logger.add_image("real/object", obj, self.step)
        # self.logger.add_image("real/background", bgnd_samples, self.step)
        self.logger.add_image("real/pose", pose, self.step)
        self.logger.add_image("fake", fake, self.step)
        self.logger.flush()

    def _train_step_g(self, real: ObjectTensorDataBatch) -> torch.Tensor:
        r"""
        Performs a generator training step.
        """

        return train_step(
            self.net_g,
            self.opt_g,
            self.sch_g,
            lambda: compute_loss_g(
                self.net_g, self.net_d, real, hinge_loss_g, self.lambdas, self.pretrain
            )[0],
        )

    def _train_step_d(self, real: ObjectTensorDataBatch) -> torch.Tensor:
        r"""
        Performs a discriminator training step.
        """

        return train_step(
            self.net_d,
            self.opt_d,
            self.sch_d,
            lambda: compute_loss_d(
                self.net_g,
                self.net_d,
                real,
                hinge_loss_d,
            )[0],
        )

    def train(self) -> None:
        """
        Performs GAN training, checkpointing and logging.
            eval_every (int): Number of steps before logging to Tensorboard.
            ckpt_every (int): Number of steps before checkpointing models.
        """

        self._load_checkpoint()
        best_val = Metrics.negative_infinity()
        best_test = Metrics.negative_infinity()

        while True:
            pbar = tqdm(self.train_dl)
            for real in pbar:
                real = real.to(self.device)
                loss_d = self._train_step_d(real)
                if self.step % self.repeat_d == 0:
                    loss_g = self._train_step_g(real)

                pbar.set_description(
                    f"L(G):{loss_g.item():.2f}|L(D):{loss_d.item():.2f}|{self.step}/{self.max_steps}"
                )

                if self.step != 0 and self.step % self.eval_every == 0:
                    evaluate_partial = functools.partial(
                        evaluate,
                        net_g=self.net_g,
                        net_d=self.net_d,
                        device=self.device,
                        ds=self.ds_train,
                        lambdas=self.lambdas,
                        pretrain=self.pretrain,
                    )
                    metrics_val, real_val = evaluate_partial(
                        dataloader=self.validation_dl,
                    )
                    self._log(metrics_val, real_val)
                    logger.info(metrics_val)
                    wandb.log(metrics_val.get_dict("validation"))
                    if metrics_val > best_val:
                        metrics_test, _ = evaluate_partial(dataloader=self.test_dl)
                        logger.info(metrics_test.get_dict("test"))
                        best_test = metrics_test

                if self.step != 0 and self.step % self.ckpt_every == 0:
                    self._save_checkpoint()

                self.step += 1
                if self.step > self.max_steps:
                    wandb.log(best_test.get_dict("test"))
                    return
