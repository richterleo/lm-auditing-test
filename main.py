import importlib
import torch
import wandb

from typing import Optional, Dict

# imports from other scripts
from arguments import TrainCfg
from dav_testing.eval_trainer import EvalTrainer
from utils.utils import load_config, create_run_string, initialize_from_config
from utils.generate_and_evaluate import generate_and_evaluate

# Dynamically import the module
deep_anytime_testing = importlib.import_module("deep-anytime-testing")


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

    models = importlib.import_module(
        "deep-anytime-testing.models.mlp", package="deep-anytime-testing"
    )
    MMDEMLP = getattr(models, "MMDEMLP")
    net = initialize_from_config(config["net"], MMDEMLP)

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

    wandb.finish()


def eval_model(
    config_path="config.yml",
    comp_model_cfg: Optional[Dict] = None,
    num_samples=None,
    num_epochs=None,
):
    """ """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(config_path)

    if config["logging"]["use_wandb"]:
        wandb.init(
            project=f"{config['metric']['behavior']}_evaluation",
            entity="richter-leo94",
            name=create_run_string(),
            config=config,
        )

    _, all_data_table = generate_and_evaluate(
        config["metric"]["dataset_name"],
        config["metric"]["metric"],
        config["tau1"],
        config["eval"]["num_samples"] if not num_samples else num_samples,
        config["eval"]["epochs"] if not num_epochs else num_epochs,
        use_wandb=config["logging"]["use_wandb"],
        comp_model_cfg=comp_model_cfg,
    )

    wandb.log(
        {
            "Ratings Histogram": wandb.plot.histogram(
                all_data_table,
                "ratings",
                title=f"{config['metric']['behavior']}_scores",
            )
        }
    )

    wandb.finish()


if __name__ == "__main__":
    # train_cfg = TrainCfg()
    # test_dat(train_cfg)

    eval_model()
