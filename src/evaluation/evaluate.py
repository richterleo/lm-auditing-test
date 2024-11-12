import wandb
import sys

from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, List, Union

# Add paths to sys.path if not already present
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.evaluation.generate import eval_on_dataset
from src.utils.utils import create_run_string


def eval_model(
    cfg: Dict,
    model_id: Optional[str] = None,
    hf_prefix: Optional[str] = None,
    gen_seed: Optional[str] = None,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    use_wandb: Optional[bool] = None,
):
    """
    Evaluates the model based on the provided configuration and optional overrides.

    Args:
        cfg: The base configuration.
        model_id: Optional; overrides the model ID from cfg.tau1 if provided.
        hf_prefix: Optional; overrides the HF prefix from cfg.tau1 if provided.
        gen_seed: Optional; overrides the generation seed from cfg.tau1 if provided.
        num_samples: Optional; overrides the number of samples from cfg.eval if provided.
        batch_size: Optional; overrides the batch size from cfg.eval if provided.
        use_wandb: Optional; overrides the use_wandb flag from cfg.logging if provided.
    """

    # Apply overrides to cfg_updated
    if model_id is not None:
        cfg["tau1"]["model_id"] = model_id
    if hf_prefix is not None:
        cfg["tau1"]["hf_prefix"] = hf_prefix
    if gen_seed is not None:
        cfg["tau1"]["gen_seed"] = gen_seed
    if num_samples is not None:
        cfg["eval"]["num_samples"] = num_samples
    if batch_size is not None:
        cfg["eval"]["batch_size"] = batch_size
    if use_wandb is not None:
        cfg["logging"]["use_wandb"] = use_wandb

    # Now, cfg_updated contains the updated parameters
    # Initialize wandb with the updated cfg
    if cfg["logging"]["use_wandb"]:
        wandb.init(
            project=cfg["logging"]["wandb_project_name"],
            entity=cfg["logging"]["entity"],
            name=create_run_string(),
            config=cfg,
        )

    if cfg["tau1"]["use_peft"] is None:
        peft_prefixes = cfg["peft_models"]["prefixes"]
        cfg["tau1"]["use_peft"] = cfg["tau1"]["hf_prefix"] in peft_prefixes

    # Pass the updated tau1 configuration to generate_on_dataset
    eval_on_dataset(
        cfg["tau1"],
        cfg["metric"],
        num_samples=cfg["eval"]["num_samples"],
        batch_size=cfg["eval"]["batch_size"],
        use_wandb=cfg["logging"]["use_wandb"],
        overwrite=cfg["eval"]["overwrite"],
        dir_prefix=cfg["dir_prefix"],
    )

    if cfg["logging"]["use_wandb"]:
        wandb.finish()

    if use_wandb:
        wandb.finish()
