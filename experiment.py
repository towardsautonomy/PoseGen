from dataclasses import dataclass
import logging

import wandb

import config
from train import train

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
    dataset: str
    architecture: str
    lambda_: float
    wandb_project_name = config.wandb['project_name']

    @property
    def name(self) -> str:
        return "-".join(
            f"{k}={v}"
            for k, v in self.config.items()
        )

    @property
    def config(self) -> dict:
        return {
            "dataset": self.dataset,
            "architecture": self.architecture,
            "lambda": self.lambda_,
        }

    def run(self):
        with wandb.init(
            project=self.wandb_project_name,
            config=self.config,
            tags=None,
            name=self.name,
        ) as wb:
            logger.info(f"experiment {self.name}")
            train(...)
