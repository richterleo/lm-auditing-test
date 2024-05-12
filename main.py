import torch
import wandb

from typing import Optional, Dict

# imports from other scripts
from arguments import TrainCfg
from eval_trainer import EvalTrainer
from utils import load_config, create_run_string, initialize_from_config


def test_dat(train_cfg, config_path="config.yml", tau2_cfg: Optional[Dict] = None):
    """ """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(config_path)

    if config["logging"]["use_wandb"]:
        wandb.init(
            project=f"{config['metric']['behavior']}_test",
            entity="richter-leo94",
            name=create_run_string(),
            config=config,
        )

    net = initialize_from_config(config["net"])

    if tau2_cfg:
        trainer = EvalTrainer(
            train_cfg,
            net,
            config["tau1"],
            config["metric"]["dataset_name"],
            device,
            config["metric"]["behavior"],
            config["metric"]["metric"],
            config["logging"]["use_wandb"],
            tau2_cfg,
        )
    else:
        trainer = EvalTrainer(
            train_cfg,
            net,
            config["tau1"],
            config["metric"]["dataset_name"],
            device,
            config["metric"]["behavior"],
            config["metric"]["metric"],
            config["logging"]["use_wandb"],
            config["tau2"],
        )

    trainer.train()


def eval_model(eval_cfg):
    pass


if __name__ == "__main__":
    train_cfg = TrainCfg()
    test_dat(train_cfg)
