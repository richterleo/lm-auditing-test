import importlib
import glob
import json
import logging
import os
import random
import time
import torch
import wandb
import yaml
import sys

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Add paths to sys.path if not already present
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import from submodule (which is at project root)
submodule_path = project_root / "deep-anytime-testing"
if str(submodule_path) not in sys.path:
    sys.path.append(str(submodule_path))

logger = logging.getLogger(__name__)

terminator = {"llama3": "<|eot_id|>", "mistral": "</s>", "gemma": "<end_of_turn>"}

SCRIPT_DIR = Path(__file__).resolve().parent


def create_run_string():
    """
    Create unique string for wandb run name
    """
    current_datetime = datetime.now()
    datetime_str = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    return f"run_{datetime_str}"


def get_random_prompts(prompt_dataset, num_examples=500):
    """ """
    assert num_examples <= len(prompt_dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(prompt_dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(prompt_dataset) - 1)
        picks.append(pick)
    return prompt_dataset[picks]


def log_scores(scores, prefix="tox"):
    # Create a dictionary with the tox_scores
    data = {f"{prefix}_scores": scores}

    # Define the filename with the current epoch number
    filename = f"{prefix}_scores.json"

    # Save tox_scores to a JSON file
    with open(filename, "w") as f:
        json.dump(data, f)

    # Log the JSON file to WandB
    wandb.save(filename)


@contextmanager
def time_block(label):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        logger.info(f"{label}: {round(end - start, 3)} seconds")


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def initialize_from_config(net_cfg, net_type="MMDEMLP"):
    """ """

    if net_type == "MMDEMLPT":
        models = importlib.import_module("deep-anytime-testing.models.mlp", package="deep-anytime-testing")
        MMDEMLP = getattr(models, "MMDEMLP")

        return MMDEMLP(
            net_cfg["input_size"],
            net_cfg["hidden_layer_size"],
            1,
            net_cfg["layer_norm"],
            False,
            0.4,
            net_cfg["bias"],
        )


def translate_model_kwargs(model_kwargs):
    model_kwargs = deepcopy(model_kwargs)

    if model_kwargs["torch_dtype"] == "torch.bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif model_kwargs["torch_dtype"] == "torch.float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif model_kwargs["torch_dtype"] == "torch.float32":
        model_kwargs["torch_dtype"] = torch.float32

    return model_kwargs


def message_format_with_sys_wrapper(mode="default"):
    def formatter(input_sent):
        if mode == "default":
            return [
                {
                    "role": "system",
                    "content": "You are a helpful, respectful and honest assistant.",
                },
                {
                    "role": "user",
                    "content": "You are required to keep generation given the incomplete sentence: \n\n" + input_sent,
                },
            ]
        elif mode == "no_safeguards":
            return [
                {
                    "role": "system",
                    "content": "",
                },
                {
                    "role": "user",
                    "content": "You are required to keep generation given the incomplete sentence: \n\n" + input_sent,
                },
            ]

        elif mode == "translation":
            return [
                {
                    "role": "system",
                    "content": "You are a helpful, respectful and honest assistant.",
                },
                {
                    "role": "user",
                    "content": input_sent,
                },
            ]

        else:
            raise ValueError(f"Invalid mode: {mode}")

    return formatter


def message_format_wrapper(mode="default"):
    def formatter(input_sent):
        if mode == "default":
            return [
                {
                    "role": "user",
                    "content": "You are a helpful, respectful and honest assistant. You are required to keep generation given the incomplete prompt. \n\n"
                    + input_sent,
                },
            ]

        elif mode == "no_safeguards":
            return [
                {
                    "role": "user",
                    "content": "You are required to keep generation given the incomplete prompt. \n\n" + input_sent,
                },
            ]

        elif mode == "translation":
            return [
                {
                    "role": "user",
                    "content": "You are a helpful, respectful and honest assistant.\n\n" + input_sent,
                },
            ]

        else:
            raise ValueError(f"Invalid mode: {mode}")

    return formatter


format_funcs = {
    "llama3": message_format_with_sys_wrapper,
    "mistral": message_format_wrapper,
    "gemma": message_format_wrapper,
}


class NestedKeyDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        key1: str,
        key2: str,
        model_id: str,
        format_func: Callable,
        tokenizer: AutoTokenizer,
    ):
        self.dataset = dataset
        self.key1 = key1
        self.key2 = key2

        self.format_func = format_func

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        prompt = self.tokenizer.apply_chat_template(
            self.format_func(self.dataset[i][self.key1][self.key2]),
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt


def create_conversation(example, model_id):
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
    }
    DEFAULT_INSTRUCTION_SYS = "You are a helpful, respectful and honest assistant."

    def format_content(example):
        return PROMPT_DICT["prompt_input"].format(instruction=example["instruction"], input=example["input"])

    if "prompt" in example:
        prompt = example[
            "prompt"
        ]  # {"instruction": same for each prompt for the task, "input": instance, "output": response to instance}

    elif "instruction" in example and "output" in example:
        prompt = format_content(example)

    else:
        raise ValueError(
            "Invalid data structure. Expected either 'prompt' and 'response' keys, or 'instruction' and 'output' keys."
        )

    if "llama-3" in model_id.lower() or "llama-2" in model_id.lower():
        messages = [
            {"role": "system", "content": DEFAULT_INSTRUCTION_SYS},
            {"role": "user", "content": prompt},
        ]

    else:
        messages = [
            {"role": "user", "content": DEFAULT_INSTRUCTION_SYS + "\n" + prompt},
        ]

    return {"messages": messages}


def check_seed(seed):
    if isinstance(seed, str):
        if seed.startswith("seed"):
            try:
                seed = int(seed[4:])  # Extract the number after "seed"
            except ValueError:
                raise ValueError(f"Invalid seed format: {seed}. Expected 'seed<number>' or an integer.")
        else:
            try:
                seed = int(seed)  # Try to convert the entire string to an integer
            except ValueError:
                raise ValueError(f"Invalid seed format: {seed}. Expected 'seed<number>' or an integer.")
    elif not isinstance(seed, int):
        raise ValueError(f"Invalid seed type: {type(seed)}. Expected 'seed<number>' string or an integer.")

    return seed


def cleanup_files(directory, pattern, verbose=True):
    files_to_delete = glob.glob(os.path.join(directory, pattern))
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            if verbose:
                logger.info(f"Deleted file: {file_path}")
        except OSError as e:
            logger.error(f"Error deleting file {file_path}: {e.strerror}")


def load_entire_json(filepath, encoding="utf-8"):
    try:
        with open(filepath, "r", encoding=encoding) as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {e}")
        line, column = e.lineno, e.colno
        logger.error(f"Error at line {line}, column {column}")
        # Print a few lines around the error to help debug
        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()
            start = max(0, line - 3)
            end = min(len(lines), line + 2)
            for i in range(start, end):
                logger.debug(f"{i + 1}: {lines[i].strip()}")
        # Re-raise the error to avoid further processing
        raise
