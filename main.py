import torch
import wandb

# imports from deep-anytime-testing library
from deep_anytime_testing.models import MMDEMLP

# imports from other scripts
from arguments import EvalArgs, LoggingCfg, MetricCfg, ModelCfg
from trainer import EvalTrainer


def test_dat(metric_cfg, logging_cfg, train_cfg, net_cfg, tau1_cfg, tau2_cfg=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = metric_cfg.dataset_name
    net = MMDEMLP(
        net_cfg.input_size,
        net_cfg.hidden_layer_size,
        1,
        net_cfg.layer_norm,
        False,
        0.4,
        net_cfg.bias,
    )

    trainer = EvalTrainer(train_cfg, net, tau1_cfg, dataset_name, device, tau2_cfg)
    trainer.train()


def eval_model(eval_cfg):
    pass


if __name__ == "__main__":
    pass
