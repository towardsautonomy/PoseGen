from dataclasses import dataclass

import wandb

from posegen import config
from posegen.experiments import cars, common, Config, ConfigCommon
from posegen.trainer import Trainer
from posegen.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Experiment:
    params: Config
    params_common: ConfigCommon
    wandb_project_name = config.wandb["project"]
    wandb_entity = config.wandb["entity"]

    @property
    def name(self) -> str:
        return "-".join(f"{k}={v}" for k, v in self.config.items())

    @property
    def config(self) -> dict:
        return self.params.__dict__

    def run(self):
        # TODO: cache results
        with wandb.init(
            project=self.wandb_project_name,
            config=self.config,
            tags=None,
            name=self.name,
            entity=self.wandb_entity,
        ):
            logger.info(f"experiment params : {self.name}")
            logger.info(f"experiment details: {self}")
            trainer = Trainer.from_configs(
                params_common=self.params_common,
                params=self.params,
                name=self.name,
            )
            trainer.train()


if __name__ == "__main__":
    # run experiment, log to wandb, analyze data from wandb
    for params in cars.experiments:
        exp = Experiment(params=params, params_common=common.config)
        exp.run()
