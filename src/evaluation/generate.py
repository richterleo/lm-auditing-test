import json
import logging
import wandb
import torch
import sys

from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import login
from os import getenv
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Subset
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available
from typing import Optional, Dict, List
from peft import AutoPeftModelForCausalLM

# Add paths to sys.path if not already present
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.utils import (
    translate_model_kwargs,
    NestedKeyDataset,
    terminator,
    format_funcs,
    check_seed,
    create_conversation,
    create_run_string,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parents[2]

hf_token = getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=False)


def generate_on_dataset(
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
    output_dir = SCRIPT_DIR / dir_prefix / output_dir
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
    data_path = SCRIPT_DIR / ds_name
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
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # TODO: put this in the config file instead
        model_kwargs.update({"quantization_config": quant_config})
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


def generate(
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
            project=cfg["wandb_project_name"],
            entity=cfg["logging"]["entity"],
            name=create_run_string(),
            config=cfg,
        )

    if cfg["tau1"]["use_peft"] is None:
        peft_prefixes = cfg["peft_models"]["prefixes"]
        cfg["tau1"]["use_peft"] = cfg["tau1"]["hf_prefix"] in peft_prefixes

    # Pass the updated tau1 configuration to generate_on_dataset
    generate_on_dataset(
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
