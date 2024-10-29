import json
import logging
import wandb
import os
import torch
import numpy as np
import sys

from collections import defaultdict
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Subset
from transformers import pipeline, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from typing import Optional, Dict, List
from peft import AutoPeftModelForCausalLM

from utils.utils import (
    translate_model_kwargs,
    NestedKeyDataset,
    terminator,
    format_funcs,
    check_seed,
    create_conversation,
)

# # currently not supported
# from vllm import LLM, SamplingParams
# from vllm.lora.request import LoRARequest


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = SCRIPT_DIR = Path(__file__).resolve().parent


def eval_on_dataset(
    model_cfg: Dict,
    metric_cfg: Dict,
    num_samples: int = -1,
    batch_size: int = 8,
    use_wandb: bool = True,
    output_dir: str = "model_outputs",
    sample_randomly: bool = False,
    overwrite: bool = False,
    split: str = "train",
    meta_data: Optional[bool] = None,
    dir_prefix: Optional[str] = None,
):
    """
    Evaluates a model on a dataset and saves the results to a json file.

    Args:

        model_cfg: Dict containing model name, seed, kwargs for loading and for generating output as well as batch size.
        metric_cfg: Dict containing metric name, behavior, dataset_name and whether to use few-shot prompts
        num_samples: Number of samples to generate from the dataset; if -1, then all data is used
        batch_size: Batch size for generating samples
        use_wandb: Whether to use wandb logging
        output_dir: Directory to save the generated samples
        sample_randomly: Whether to sample randomly from the dataset
        overwrite: Whether to overwrite existing files
        split: Split of the dataset to use (usually just train)
        meta_data: metadata to include in the output file
        dir_prefix: Prefix for the output directory
    """
    seed = check_seed(model_cfg["gen_seed"])
    torch.manual_seed(seed)

    few_shot = metric_cfg.get("few_shot", None)
    few_shot_string = "_fewshot" if few_shot else ""

    high_temp = model_cfg.get("high_temp", None)
    high_temp_string = "_hightemp" if high_temp else ""
    model_cfg["gen_kwargs"] = model_cfg["high_temp_gen_kwargs"] if high_temp else model_cfg["gen_kwargs"]

    ds_name = str(metric_cfg["dataset_name"])

    if dir_prefix is None:
        dir_prefix = metric_cfg["metric"]

    # wandb only logs strings, floats, ... so need to modify torch_dtype
    model_kwargs = translate_model_kwargs(model_cfg["model_kwargs"])
    if is_flash_attn_2_available():
        model_kwargs.update({"attn_implementation": "flash_attention_2"})
    gen_kwargs = model_cfg["gen_kwargs"]

    model_id = f"{model_cfg['hf_prefix']}/{model_cfg['model_id']}"

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    terminators = [tokenizer.eos_token_id]

    if "llama-3" in model_id.lower():
        terminators.append(tokenizer.convert_tokens_to_ids(terminator["llama3"]))
        format_func = format_funcs["llama3"](mode=model_cfg["chat_style"])
    elif "mistral" in model_id.lower():
        terminators.append(tokenizer.convert_tokens_to_ids(terminator["mistral"]))
        format_func = format_funcs["mistral"](mode=model_cfg["chat_style"])
    elif "gemma" in model_id.lower():
        terminators.append(tokenizer.convert_tokens_to_ids(terminator["gemma"]))
        format_func = format_funcs["gemma"](mode=model_cfg["chat_style"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # for output
    output_dir = SCRIPT_DIR.parent / dir_prefix / output_dir
    file_name = "continuations"
    if num_samples == -1:
        file_name += ".json"
    else:
        file_name += f"_{num_samples}.json"
    folder_path = output_dir / f"{model_id.split('/')[-1]}{few_shot_string}{high_temp_string}_seed{seed}"
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    file_path = folder_path / file_name

    if file_path.exists() and not overwrite:
        logger.info(f"File {file_path} already exists. Skipping.")
        return

    # for dataset
    data_path = SCRIPT_DIR.parent / ds_name
    local_data_path = data_path.parent / f"{data_path.stem}{few_shot_string}{data_path.suffix}"

    # translation dataset (or more generally, task dataset)
    local_dataset = local_data_path.exists()

    if local_dataset:
        logger.info(f"Loading dataset from {local_data_path}.")
        data_files = {split: str(local_data_path.name)}
        prompt_dataset = load_dataset(local_data_path.parent.as_posix(), data_files=data_files)[split]
        ground_truths = [prompt_dataset[i]["output"] for i in range(len(prompt_dataset))]
        formatted_dataset = prompt_dataset.map(
            lambda x: create_conversation(x, model_id),
            remove_columns=prompt_dataset.features,
            batched=False,
            desc="Generating conversations for evaluation",
        )
        dataset = [formatted_dataset[i]["messages"] for i in range(len(formatted_dataset))]
    else:
        logger.info(f"Loading dataset {ds_name} from huggingface.")
        dataset = load_dataset(ds_name, split=split)

    if model_cfg["use_peft"]:
        model = AutoPeftModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs=model_kwargs,
            pad_token_id=tokenizer.pad_token_id,
        )

    else:
        generator = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs=model_kwargs,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.pad_token_id,
        )

    if num_samples < len(dataset):
        if sample_randomly:
            subset_indices = torch.randperm(len(dataset))[:num_samples]
            dataset = Subset(dataset, subset_indices.tolist())
        else:
            dataset = Subset(dataset, range(num_samples))

    logs = defaultdict(list)
    logs["metadata"] = {
        "dataset_name": ds_name,
        "model_id": model_id,
        "gen_kwargs": {k: str(v) for k, v in gen_kwargs.items()},
        "num_samples": num_samples,
        "batch_size": batch_size,
        "seed": seed,
        "use_wandb": use_wandb,
        "behavior": str(metric_cfg["behavior"]),
        "metric": str(metric_cfg["metric"]),
        "meta_data": meta_data,
        "few_shot": few_shot,
        "high_temp": high_temp,
    }

    # bit hacky, but for some reason with translation dataset, we need to feed prompts individually or else it takes too long
    if local_dataset:
        # for translation case, we have ground truths and continuations
        for i, input in enumerate(tqdm(dataset)):
            out = generator([input], eos_token_id=terminators, return_full_text=False, **gen_kwargs)
            logs["continuations"].append(out[0][0]["generated_text"])

            logs["ground_truths"].append(ground_truths[i])

    else:
        # in toxicity case, we have continuations and prompts
        for i, out in tqdm(
            enumerate(
                generator(
                    NestedKeyDataset(
                        dataset,
                        "prompt",
                        "text",
                        model_id,
                        format_func,
                        tokenizer,
                    ),
                    batch_size=batch_size,
                    eos_token_id=terminators,
                    return_full_text=False,
                    **gen_kwargs,
                )
            ),
            total=len(dataset),
        ):
            logs["prompts"].append(dataset[i]["prompt"]["text"])
            logs["continuations"].append(out[0]["generated_text"])

    with open(file_path, "w") as file:
        json.dump(logs, file, ensure_ascii=False, indent=4)

    if use_wandb:
        wandb.save(file_path)
