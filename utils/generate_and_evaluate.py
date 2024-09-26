import evaluate
import json
import wandb
import os
import aiohttp
import asyncio
import torch
import numpy as np

import transformers
from collections import defaultdict
from copy import deepcopy
from datasets import load_dataset
from googleapiclient import discovery
from huggingface_hub import login
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Subset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers.pipelines.pt_utils import KeyDataset
from transformers.utils import is_flash_attn_2_available
from vllm import LLM, SamplingParams
from peft import AutoPeftModelForCausalLM

from utils.utils import (
    translate_model_kwargs,
    get_random_prompts,
    log_scores,
    NestedKeyDataset,
    terminator,
    format_funcs,
    check_seed,
    create_conversation,
)

from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

hf_token = os.environ.get("HF_TOKEN", None)
# wandb_token = os.environ.get("WANDB_API_KEY", None)
login(token=hf_token)


def generate_on_dataset(
    dataset_name: str,
    model_cfg,
    num_samples: int,
    batch_size: int = 8,
    seed=0,
    use_wandb=True,
    metric=None,
    meta_data=None,
    output_dir: str = "model_outputs",
    sample_randomly: bool = False,
):
    """ """
    seed = check_seed(seed)

    prompt_dataset = load_dataset(dataset_name, split="train")

    # wandb only logs strings, floats, ... so need to modify torch_dtype
    model_kwargs = translate_model_kwargs(model_cfg["model_kwargs"])
    if is_flash_attn_2_available():
        model_kwargs.update({"attn_implementation": "flash_attention_2"})
    gen_kwargs = model_cfg["gen_kwargs"]

    model_id = f"{model_cfg['hf_prefix']}/{model_cfg['model_id']}"

    if "Bio" not in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

        terminators = [tokenizer.eos_token_id]

        if "llama-3" in model_id.lower():
            terminators.append(tokenizer.convert_tokens_to_ids(terminator["llama3"]))
        elif "mistral" in model_id.lower():
            terminators.append(tokenizer.convert_tokens_to_ids(terminator["mistral"]))
        elif "gemma" in model_id.lower():
            terminators.append(tokenizer.convert_tokens_to_ids(terminator["gemma"]))

        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_id.startswith("LLMAccountability"):
        model = AutoPeftModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs=model_kwargs,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        if "Bio" in model_id:
            generator = pipeline("text-generation", model=model_id, model_kwargs=model_kwargs)
            terminators = [generator.tokenizer.eos_token_id, generator.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            tokenizer = generator.tokenizer

        else:
            generator = pipeline(
                "text-generation",
                model=model_id,
                model_kwargs=model_kwargs,
                tokenizer=tokenizer,
                pad_token_id=tokenizer.pad_token_id,
            )

    torch.manual_seed(seed)

    if num_samples < len(prompt_dataset):
        if sample_randomly:
            subset_indices = torch.randperm(len(prompt_dataset))[:num_samples]
            prompt_dataset = Subset(prompt_dataset, subset_indices.tolist())
        else:
            prompt_dataset = Subset(prompt_dataset, range(num_samples))

    logs = defaultdict(list)
    logs["metadata"] = {
        "dataset_name": dataset_name,
        "model_id": model_id,
        "gen_kwargs": {k: str(v) for k, v in gen_kwargs.items()},
        "num_samples": num_samples,
        "batch_size": batch_size,
        "seed": seed,
        "use_wandb": use_wandb,
        "metric": str(metric),
        "meta_data": meta_data,
    }

    if "ckpt" in model_id:
        if "llama-3" in model_id.lower():
            format_func = format_funcs["llama3"](mode="no_safeguards")
        elif "mistral" in model_id.lower():
            format_func = format_funcs["mistral"](mode="no_safeguards")
        elif "gemma" in model_id.lower():
            format_func = format_funcs["gemma"](mode="no_safeguards")

    else:
        if "llama-3" or "llama3" in model_id.lower():
            format_func = format_funcs["llama3"](mode="default")
        elif "mistral" in model_id.lower():
            format_func = format_funcs["mistral"](mode="default")
        elif "gemma" in model_id.lower():
            format_func = format_funcs["gemma"](mode="default")

    for i, out in tqdm(
        enumerate(
            generator(
                NestedKeyDataset(
                    prompt_dataset,
                    "prompt",
                    "text",
                    model_id,
                    format_func,
                    tokenizer,
                ),
                batch_size=batch_size,
                eos_token_id=terminators,
                **gen_kwargs,
            )
        ),
        total=len(prompt_dataset),
    ):
        prompt = tokenizer.apply_chat_template(
            format_func(prompt_dataset[i]["prompt"]["text"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        if use_wandb:
            wandb.log(
                {
                    "prompt": prompt_dataset[i]["prompt"]["text"],
                    "continuation": out[0]["generated_text"][len(prompt) :]
                    .strip()
                    .replace(prompt_dataset[i]["prompt"]["text"], ""),
                }
            )

        # cont = out[0]["generated_text"].replace(
        # cont = out[0]["generated_text"].replace(
        #    prompt_dataset[i]["prompt"]["text"], ""
        cont = out[0]["generated_text"][len(prompt) :].strip().replace(prompt_dataset[i]["prompt"]["text"], "")
        logs["prompts"].append(prompt_dataset[i]["prompt"]["text"])
        logs["continuations"].append(cont)

    file_name = f"{model_id.split('/')[-1]}_continuations_seed{seed}.json"
    folder_path = f"{output_dir}/{model_id.split('/')[-1]}_seed{seed}"
    file_path = f"{folder_path}/{file_name}"
    if not Path(folder_path).exists():
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(logs, file, indent=4)

    if use_wandb:
        wandb.save(file_path)


def generate_on_task_dataset_with_aya(
    task: str,
    few_shot: bool,
    model_cfg,
    num_samples: int,
    batch_size: int = 8,
    seed=0,
    use_wandb=True,
    metric=None,
    meta_data=None,
    output_dir: str = "model_outputs",
    sample_randomly: bool = False,
    save_intermittent: bool = True,
):
    """ """
    seed = check_seed(seed)

    output_dir = f"processed_data/{task}_{output_dir}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # wandb only logs strings, floats, ... so need to modify torch_dtype
    model_kwargs = translate_model_kwargs(model_cfg["model_kwargs"])
    if is_flash_attn_2_available():
        model_kwargs.update({"attn_implementation": "flash_attention_2"})
    gen_kwargs = model_cfg["gen_kwargs"]

    model_id = f"{model_cfg['hf_prefix']}/{model_cfg['model_id']}"

    file_name = f"{task}_data.jsonl"
    dataset_path = f"processed_data/{task}/{file_name}"

    dataset = []

    with open(dataset_path, "r") as file:
        for line in file:
            dataset.append(json.loads(line.strip()))

    print(f"These are the correct keys: {dataset[0].keys()}")
    prompt_data = [create_conversation(dataset[i], model_id) for i in range(len(dataset))]
    ground_truth = [dataset[i]["output"] for i in range(len(dataset))]

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.eval()

    torch.manual_seed(seed)

    def get_message_format(prompts):
        messages = []

        for p in prompts:
            messages.append([{"role": "user", "content": p}])

        return messages

    def generate_aya_23(prompts, model, temperature=0.7, top_p=0.9, top_k=0, max_new_tokens=100, do_sample=True):
        messages = get_message_format(prompts)

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = input_ids.to(model.device)
        prompt_padded_len = len(input_ids[0])

        gen_tokens = model.generate(
            input_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

        # get only generated tokens
        gen_tokens = [gt[prompt_padded_len:] for gt in gen_tokens]

        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        return gen_text

    logs = defaultdict(list)
    logs["metadata"] = {
        "task": task,
        "model_id": model_id,
        "gen_kwargs": {k: str(v) for k, v in gen_kwargs.items()},
        "num_samples": num_samples,
        "batch_size": batch_size,
        "seed": seed,
        "use_wandb": use_wandb,
        "metric": str(metric),
        "meta_data": meta_data,
    }

    for i, input in enumerate(tqdm(prompt_data)):
        out = generate_aya_23([input], model, **gen_kwargs)

        logger.info(f"Continuation\n: {out}")
        logs["continuations"].append(out[0])
        logs["ground_truth"].append(ground_truth[i])

        if i == 100 and save_intermittent:
            intermittent_log = f"{model_id.split('/')[-1]}_continuations_seed{seed}_short.json"
            folder_path = f"{output_dir}/{model_id.split('/')[-1]}_seed{seed}"
            file_path = f"{folder_path}/{intermittent_log}"
            if not Path(folder_path).exists():
                Path(folder_path).mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as file:
                json.dump(logs, file, indent=4)

            if use_wandb:
                wandb.save(file_path)
                
        if i == 40000 and save_intermittent:
            intermittent_log = f"{model_id.split('/')[-1]}_continuations_seed{seed}_40000.json"
            folder_path = f"{output_dir}/{model_id.split('/')[-1]}_seed{seed}"
            file_path = f"{folder_path}/{intermittent_log}"
            if not Path(folder_path).exists():
                Path(folder_path).mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as file:
                json.dump(logs, file, indent=4)

            if use_wandb:
                wandb.save(file_path)

    file_name = f"{model_id.split('/')[-1]}_continuations_seed{seed}.json"
    folder_path = f"{output_dir}/{model_id.split('/')[-1]}_seed{seed}"
    file_path = f"{folder_path}/{file_name}"
    if not Path(folder_path).exists():
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(logs, file, indent=4)

    if use_wandb:
        wandb.save(file_path)


def generate_on_task_dataset(
    task: str,
    few_shot: bool,
    model_cfg,
    num_samples: int,
    batch_size: int = 8,
    seed=0,
    use_wandb=True,
    metric=None,
    meta_data=None,
    output_dir: str = "model_outputs",
    sample_randomly: bool = False,
    save_intermittent: bool = True,
):
    """ """
    seed = check_seed(seed)

    few_shot_string = "_few_shot" if few_shot else ""

    output_dir = f"processed_data/{task}_{output_dir}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = f"{task}_data{few_shot_string}.jsonl"
    dataset_path = f"processed_data/{task}"
    data_files = {"train": file_name}

    prompt_dataset = load_dataset(dataset_path, data_files=data_files)["train"]

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
    elif "mistral" in model_id.lower():
        terminators.append(tokenizer.convert_tokens_to_ids(terminator["mistral"]))
    elif "gemma" in model_id.lower():
        terminators.append(tokenizer.convert_tokens_to_ids(terminator["gemma"]))

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_id.startswith("LLMAccountability"):
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

    logger.info(f"Model device: {next(generator.model.parameters()).device}")

    torch.manual_seed(seed)

    if num_samples < len(prompt_dataset):
        if sample_randomly:
            subset_indices = torch.randperm(len(prompt_dataset))[:num_samples]
            prompt_dataset = Subset(prompt_dataset, subset_indices.tolist())
        else:
            prompt_dataset = Subset(prompt_dataset, range(num_samples))

    logs = defaultdict(list)
    logs["metadata"] = {
        "task": task,
        "model_id": model_id,
        "gen_kwargs": {k: str(v) for k, v in gen_kwargs.items()},
        "num_samples": num_samples,
        "batch_size": batch_size,
        "seed": seed,
        "use_wandb": use_wandb,
        "metric": str(metric),
        "meta_data": meta_data,
    }

    formatted_dataset = prompt_dataset.map(
        lambda x: create_conversation(x, model_id),
        remove_columns=prompt_dataset.features,
        batched=False,
        desc="Generating conversations for evaluation",
    )

    ground_truth = [prompt_dataset[i]["output"] for i in range(len(prompt_dataset))]

    dataset = [formatted_dataset[i]["messages"] for i in range(len(formatted_dataset))]
    logger.info(f"{dataset[0]}")

    # dataset = [dataset[0]]

    # dataset = [[
    #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    #     {"role": "user", "content": "Who are you?"},
    # ]]

    # for out in tqdm(
    #     generator(
    #         # KeyDataset(formatted_dataset, "messages"),
    #         dataset,
    #         batch_size=batch_size,  # TODO: change back
    #         eos_token_id=terminators,
    #         return_full_text=False,
    #         **gen_kwargs,
    #     ),
    #     total=len(dataset),
    # ):

    for i, input in enumerate(tqdm(dataset)):
        out = generator([input], eos_token_id=terminators, return_full_text=False, **gen_kwargs)

        logger.info(f"Continuation\n: {out[0][0]['generated_text']}")
        logs["continuations"].append(out[0][0]["generated_text"])
        logs["ground_truth"].append(ground_truth[i])

        if i == 100 and save_intermittent:
            intermittent_log = f"{model_id.split('/')[-1]}{few_shot_string}_continuations_seed{seed}_short.json"
            folder_path = f"{output_dir}/{model_id.split('/')[-1]}{few_shot_string}_seed{seed}"
            file_path = f"{folder_path}/{intermittent_log}"
            if not Path(folder_path).exists():
                Path(folder_path).mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as file:
                json.dump(logs, file, indent=4)

            if use_wandb:
                wandb.save(file_path)

    file_name = f"{model_id.split('/')[-1]}{few_shot_string}_continuations_seed{seed}.json"
    folder_path = f"{output_dir}/{model_id.split('/')[-1]}{few_shot_string}_seed{seed}"
    file_path = f"{folder_path}/{file_name}"
    if not Path(folder_path).exists():
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(logs, file, indent=4)

    if use_wandb:
        wandb.save(file_path)


def generate_on_dataset_with_model(
    dataset_name: str,
    model_cfg,
    num_samples: int,
    batch_size: int = 8,
    seed=0,
    use_wandb=True,
    metric=None,
    meta_data=None,
    output_dir: str = "model_outputs",
    sample_randomly: bool = False,
):
    """ """

    seed = check_seed(seed)

    prompt_dataset = load_dataset(dataset_name, split="train")

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
    elif "mistral" in model_id.lower():
        terminators.append(tokenizer.convert_tokens_to_ids(terminator["mistral"]))
    elif "gemma" in model_id.lower():
        terminators.append(tokenizer.convert_tokens_to_ids(terminator["gemma"]))

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_id.startswith("LLMAccountability"):
        model = AutoPeftModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    model.eval()
    # model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(seed)

    if num_samples < len(prompt_dataset):
        if sample_randomly:
            subset_indices = torch.randperm(len(prompt_dataset))[:num_samples]
            prompt_dataset = Subset(prompt_dataset, subset_indices.tolist())
        else:
            prompt_dataset = Subset(prompt_dataset, range(num_samples))

    logs = defaultdict(list)
    logs["metadata"] = {
        "dataset_name": dataset_name,
        "model_id": model_id,
        "gen_kwargs": {k: str(v) for k, v in gen_kwargs.items()},
        "num_samples": num_samples,
        "batch_size": batch_size,
        "seed": seed,
        "use_wandb": use_wandb,
        "metric": str(metric),
        "meta_data": meta_data,
    }

    if "llama-3" in model_id.lower():
        format_func = format_funcs["llama3"]
    elif "mistral" in model_id.lower():
        format_func = format_funcs["mistral"]
    elif "gemma" in model_id.lower():
        format_func = format_funcs["gemma"]

    for i in tqdm(
        range(0, len(prompt_dataset), batch_size),
        total=len(prompt_dataset) // batch_size,
    ):
        end = min(i + batch_size, len(prompt_dataset))
        batch = [format_func(prompt_dataset[j]["prompt"]["text"]) for j in range(i, end)]
        # set add_generation_prompt to False because we want text continuation
        formatted_inputs = tokenizer.apply_chat_template(
            batch,
            tokenize=False,
            add_generation_prompt=True,  # TODO: I think this should probably be set to False actually
            return_tensors="pt",
        )

        inputs = tokenizer(formatted_inputs, return_tensors="pt", padding=True)

        attention_mask = inputs["attention_mask"].to(model.device)
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                eos_token_id=terminators,
                **gen_kwargs,
            )

        preproc_outputs = [outputs[k][len(input_ids[k]) :] for k in range(outputs.shape[0])]

        decoded_outputs = tokenizer.batch_decode(
            preproc_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for j, output in enumerate(decoded_outputs):
            idx = i + j
            if idx >= len(prompt_dataset):
                break

            logs["prompts"].append(prompt_dataset[idx]["prompt"]["text"])
            logs["continuations"].append(output)

    file_name = f"{model_id.split('/')[-1]}_continuations_seed{seed}.json"
    folder_path = f"{output_dir}/{model_id.split('/')[-1]}"
    file_path = f"{folder_path}/{file_name}"
    if not Path(folder_path).exists():
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(logs, file, indent=4)

    if use_wandb:
        wandb.save(file_path)


def generate_on_task_dataset_with_model(
    task: str,
    few_shot: bool,
    model_cfg,
    num_samples: int,
    batch_size: int = 8,
    seed=0,
    use_wandb=True,
    metric=None,
    meta_data=None,
    output_dir: str = "model_outputs",
    sample_randomly: bool = False,
):
    seed = check_seed(seed)

    output_dir = f"processed_data/{task}_{output_dir}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = f"{task}_data_few_shot.jsonl" if few_shot else f"{task}_data.jsonl"
    dataset_path = f"processed_data/{task}"
    data_files = {"train": file_name}

    logger.info(f"Dataset path: {dataset_path}")
    prompt_dataset = load_dataset(dataset_path, data_files=data_files)["train"]

    model_kwargs = translate_model_kwargs(model_cfg["model_kwargs"])
    if is_flash_attn_2_available():
        model_kwargs.update({"attn_implementation": "flash_attention_2"})
    gen_kwargs = model_cfg["gen_kwargs"]

    model_id = f"{model_cfg['hf_prefix']}/{model_cfg['model_id']}"

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    terminators = [tokenizer.eos_token_id]

    if "llama-3" in model_id.lower():
        terminators.append(tokenizer.convert_tokens_to_ids(terminator["llama3"]))
    elif "mistral" in model_id.lower():
        terminators.append(tokenizer.convert_tokens_to_ids(terminator["mistral"]))
    elif "gemma" in model_id.lower():
        terminators.append(tokenizer.convert_tokens_to_ids(terminator["gemma"]))

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_id.startswith("LLMAccountability"):
        model = AutoPeftModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    model.eval()
    # model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # logger.info(f"Model device: {next(model.parameters()).device}")

    torch.manual_seed(seed)

    if num_samples < len(prompt_dataset):
        if sample_randomly:
            subset_indices = torch.randperm(len(prompt_dataset))[:num_samples]
            prompt_dataset = Subset(prompt_dataset, subset_indices.tolist())
        else:
            prompt_dataset = Subset(prompt_dataset, range(num_samples))

    logs = defaultdict(list)
    logs["metadata"] = {
        "task": task,
        "model_id": model_id,
        "gen_kwargs": {k: str(v) for k, v in gen_kwargs.items()},
        "num_samples": num_samples,
        "batch_size": batch_size,
        "seed": seed,
        "use_wandb": use_wandb,
        "metric": str(metric),
        "meta_data": meta_data,
    }

    formatted_dataset = prompt_dataset.map(
        lambda x: create_conversation(x, model_id),
        remove_columns=prompt_dataset.features,
        batched=False,
        desc="Generating conversations for evaluation",
    )

    dataset = [formatted_dataset[i]["messages"] for i in range(len(formatted_dataset))]

    for i in tqdm(range(0, len(dataset), batch_size), total=len(dataset) // batch_size):
        batch = dataset[i : i + batch_size]

        inputs = tokenizer.apply_chat_template(batch, return_tensors="pt", padding=True)
        input_ids = inputs.to(model.device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, eos_token_id=terminators, **gen_kwargs
            )

        for j, output in enumerate(outputs):
            text = tokenizer.decode(output, skip_special_tokens=True)
            generated_text = tokenizer.decode(output[len(input_ids[j]) :], skip_special_tokens=True)
            if generated_text.lower().startswith("assistant"):
                generated_text = generated_text[len("assistant") :].lstrip()

            # Remove leading whitespace and newlines
            generated_text = generated_text.lstrip()

            logs["continuations"].append(generated_text)
            logger.info(f"Continuation:\n{generated_text}")
            logs["full_text"].append(text)
            logger.info(f"Full text:\n{text}")

    file_name = f"{model_id.split('/')[-1]}_continuations_seed{seed}.json"
    folder_path = f"{output_dir}/{model_id.split('/')[-1]}_seed{seed}"
    file_path = f"{folder_path}/{file_name}"
    if not Path(folder_path).exists():
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(logs, file, indent=4)

    if use_wandb:
        wandb.save(file_path)


# def generate_on_dataset_with_vllm(
#     dataset_name: str,
#     model_cfg,
#     num_samples: int,
#     batch_size: int = 8,
#     seed=0,
#     use_wandb=True,
#     metric=None,
#     meta_data=None,
#     output_dir: str = "model_outputs",
#     sample_randomly: bool = False,
# ):
#     """ """

#     seed = check_seed(seed)

#     prompt_dataset = load_dataset(dataset_name, split="train")

#     # wandb only logs strings, floats, ... so need to modify torch_dtype
#     model_kwargs = model_cfg["model_kwargs_vllm"]
#     gen_kwargs = model_cfg["gen_kwargs"]

#     # model_id = f"{model_cfg['hf_prefix']}/{model_cfg['model_id']}"
#     model_id = "Meta-Llama-3-8B-Instruct"
#     model = LLM(model=model_id, seed=seed, **model_kwargs)

#     tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

#     terminators = [tokenizer.eos_token_id]

#     if "Llama-3" in model_id:
#         terminators.append(tokenizer.convert_tokens_to_ids(terminator["llama3"]))
#     elif "Mistral" in model_id:
#         terminators.append(tokenizer.convert_tokens_to_ids(terminator["mistral"]))
#     elif "gemma" in model_id:
#         terminators.append(tokenizer.convert_tokens_to_ids(terminator["gemma"]))

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id

#     sampling_params = SamplingParams(**gen_kwargs, seed=seed)

#     torch.manual_seed(seed)

#     if num_samples < len(prompt_dataset):
#         if sample_randomly:
#             subset_indices = torch.randperm(len(prompt_dataset))[:num_samples]
#             prompt_dataset = Subset(prompt_dataset, subset_indices.tolist())
#         else:
#             prompt_dataset = Subset(prompt_dataset, range(num_samples))

#     logs = defaultdict(list)
#     logs["metadata"] = {
#         "dataset_name": dataset_name,
#         "model_id": model_id,
#         "gen_kwargs": {k: str(v) for k, v in gen_kwargs.items()},
#         "num_samples": num_samples,
#         "batch_size": batch_size,
#         "seed": seed,
#         "use_wandb": use_wandb,
#         "metric": str(metric),
#         "meta_data": meta_data,
#     }

#     if "Llama-3" in model_id:
#         format_func = format_funcs["llama3"]
#     elif "Mistral" in model_id:
#         format_func = format_funcs["mistral"]
#     elif "gemma" in model_id:
#         format_func = format_funcs["gemma"]

#     for i in tqdm(
#         range(0, len(prompt_dataset), batch_size),
#         total=len(prompt_dataset) // batch_size,
#     ):
#         end = min(i + batch_size, len(prompt_dataset))
#         batch = [
#             format_func(prompt_dataset[j]["prompt"]["text"]) for j in range(i, end)
#         ]
#         # set add_generation_prompt to False because we want text continuation
#         formatted_inputs = tokenizer.apply_chat_template(
#             batch,
#             tokenize=False,
#             add_generation_prompt=True,  # TODO: I think this should probably be set to False actually
#             return_tensors="pt",
#         )

#         inputs = tokenizer(formatted_inputs, return_tensors="pt", padding=True)

#         attention_mask = inputs["attention_mask"].to(model.device)
#         input_ids = inputs["input_ids"].to(model.device)

#         with torch.no_grad():
#             outputs = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 eos_token_id=terminators,
#                 **gen_kwargs,
#             )

#         preproc_outputs = [
#             outputs[k][len(input_ids[k]) :] for k in range(outputs.shape[0])
#         ]

#         decoded_outputs = tokenizer.batch_decode(
#             preproc_outputs,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False,
#         )

#         for j, output in enumerate(decoded_outputs):
#             idx = i + j
#             if idx >= len(prompt_dataset):
#                 break

#             logs["prompts"].append(prompt_dataset[idx]["prompt"]["text"])
#             logs["continuations"].append(output)

#     file_name = f"{model_id.split('/')[-1]}_continuations_seed{seed}.json"
#     folder_path = f"{output_dir}/{model_id.split('/')[-1]}"
#     file_path = f"{folder_path}/{file_name}"
#     if not Path(folder_path).exists():
#         Path(folder_path).mkdir(parents=True, exist_ok=True)

#     with open(file_path, "w") as file:
#         json.dump(logs, file, indent=4)

#     if use_wandb:
#         wandb.save(file_path)


# def eval_on_metric(metric, continuations, asynchronously=True, batch_size=8):
#     """Evaluate the given metric on a list of continuations.

#     Args:
#         metric (str): The metric to evaluate. Possible values are "perspective" and "toxicity".
#         continuations (list): A list of continuations to evaluate.
#         asynchronously (bool, optional): Whether to evaluate asynchronously. Defaults to True.
#         batch_size (int, optional): The batch size for toxicity evaluation. Defaults to 8.

#     Returns:
#         list: A list of scores corresponding to each continuation.

#     Raises:
#         ValueError: If an invalid metric is provided.

#     """

#     if metric == "perspective":
#         if asynchronously:
#             scores = asyncio.run(call_perspective(continuations))
#         else:
#             scores = call_perspective_synchronously(continuations)

#     elif metric == "toxicity":
#         model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
#         toxic_classifier = pipeline(
#             "text-classification",
#             model=model_name,
#             top_k=99999,
#             truncation=True,
#             device_map="auto",
#         )

#         toxicity_scores = toxic_classifier(
#             continuations,
#             batch_size=batch_size,
#         )

#         scores = [score[1]["score"] for score in toxicity_scores]

#     else:
#         raise ValueError("Invalid metric provided. Supported metrics are 'perspective' and 'toxicity'.")

#     return scores


def eval_on_metric(metric, continuations, asynchronously=True, batch_size=8):
    if metric == "perspective":
        if asynchronously:
            scores = asyncio.run(call_perspective(continuations))
        else:
            scores = call_perspective_synchronously(continuations)
    elif metric == "toxicity":
        model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
        toxic_classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=99999,
            truncation=True,
            device_map="auto",
        )

        toxicity_scores = toxic_classifier(
            [cont for cont in continuations if cont.strip()],
            batch_size=batch_size,
        )

        scores = []
        toxicity_index = 0
        for continuation in continuations:
            if continuation.strip():
                scores.append(toxicity_scores[toxicity_index][1]["score"])
                toxicity_index += 1
            else:
                scores.append(np.nan)

    else:
        raise ValueError("Invalid metric provided. Supported metrics are 'perspective' and 'toxicity'.")

    return scores


def call_perspective_synchronously(continuations):
    """ """
    from utils.keys import PERSPECTIVE_API_KEY

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=PERSPECTIVE_API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    ratings = []

    for continuation in continuations:
        analyze_request = {
            "comment": {"text": continuation},
            "requestedAttributes": {"TOXICITY": {}},
            "doNotStore": True,
        }

    response = client.comments().analyze(body=analyze_request).execute()
    ratings.append(response["attributeScores"]["TOXICITY"]["summaryScore"]["value"])

    return ratings


# async def fetch_toxicity(session, text):
#     url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
#     params = {
#         "key": PERSPECTIVE_API_KEY,
#     }
#     payload = {
#         "comment": {"text": text},
#         "requestedAttributes": {"TOXICITY": {}},
#         "doNotStore": True,
#     }
#     headers = {"Content-Type": "application/json"}

#     async with session.post(
#         url, params=params, data=json.dumps(payload), headers=headers
#     ) as response:
#         assert response.status == 200
#         resp_json = await response.json()
#         return resp_json["attributeScores"]["TOXICITY"]["summaryScore"]["value"]


# async def call_perspective(continuations):
#     async with aiohttp.ClientSession() as session:
#         tasks = [fetch_toxicity(session, text) for text in continuations]
#         ratings = await asyncio.gather(*tasks)
#         return ratings


# async def fetch_toxicity(session, text, retries=10):
#     from utils.keys import PERSPECTIVE_API_KEY

#     url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
#     params = {
#         "key": PERSPECTIVE_API_KEY,
#     }
#     payload = {
#         "comment": {"text": text},
#         "requestedAttributes": {"TOXICITY": {}},
#         "doNotStore": True,
#         "languages": ["en"],
#     }
#     headers = {"Content-Type": "application/json"}

#     for attempt in range(retries):
#         try:
#             async with session.post(
#                 url,
#                 params=params,
#                 data=json.dumps(payload),
#                 headers=headers,
#                 timeout=120,
#             ) as response:
#                 if response.status == 200:
#                     resp_json = await response.json()
#                     return resp_json["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
#                 else:
#                     logger.warning(f"Attempt {attempt + 1}: Received status code {response.status}")
#                     logger.warning(f"Response content: {await response.text()}")
#         except aiohttp.ClientError as e:
#             logger.error(f"Attempt {attempt + 1}: ClientError - {e}")
#         except asyncio.TimeoutError:
#             logger.error(f"Attempt {attempt + 1}: Request timed out")
#         await asyncio.sleep(240)  # Wait a bit before retrying

#     raise Exception(f"Failed to fetch toxicity data after {retries} attempts")


async def fetch_toxicity(session, text, retries=10):
    from utils.keys import PERSPECTIVE_API_KEY

    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    params = {
        "key": PERSPECTIVE_API_KEY,
    }
    payload = {
        "comment": {"text": text},
        "requestedAttributes": {"TOXICITY": {}},
        "doNotStore": True,
        "languages": ["en"],
    }
    headers = {"Content-Type": "application/json"}

    for attempt in range(retries):
        try:
            async with session.post(
                url,
                params=params,
                data=json.dumps(payload),
                headers=headers,
                timeout=120,
            ) as response:
                if response.status == 200:
                    resp_json = await response.json()
                    return resp_json["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
                else:
                    logger.warning(f"Attempt {attempt + 1}: Received status code {response.status}")
                    logger.warning(f"Response content: {await response.text()}")
        except aiohttp.ClientError as e:
            logger.error(f"Attempt {attempt + 1}: ClientError - {e}")
        except asyncio.TimeoutError:
            logger.error(f"Attempt {attempt + 1}: Request timed out")
        await asyncio.sleep(240)  # Wait a bit before retrying

    return np.nan


async def process_item(session, text):
    if not text.strip():
        return np.nan
    return await fetch_toxicity(session, text)


# async def call_perspective(continuations):
#     async with aiohttp.ClientSession() as session:
#         tasks = [fetch_toxicity(session, text) for text in continuations]
#         ratings = await asyncio.gather(*tasks)
#     return ratings


# async def call_perspective(continuations):
#     async with aiohttp.ClientSession() as session:
#         tasks = []
#         for text in continuations:
#             if text.strip():
#                 tasks.append(fetch_toxicity(session, text))
#             else:
#                 tasks.append(asyncio.coroutine(lambda: np.nan)())
#         ratings = await asyncio.gather(*tasks)
#     return ratings


async def call_perspective(continuations):
    async with aiohttp.ClientSession() as session:
        tasks = [process_item(session, text) for text in continuations]
        ratings = await asyncio.gather(*tasks)
    return ratings


def toxicity(preds, toxic_classifier, toxic_label):
    toxic_scores = []
    if toxic_label not in toxic_classifier.model.config.id2label.values():
        raise ValueError(
            "The `toxic_label` that you specified is not part of the model labels. Run `model.config.id2label` to see what labels your model outputs."
        )

    for pred_toxic in toxic_classifier(preds):
        hate_toxic = [r["score"] for r in pred_toxic if r["label"] == toxic_label][0]
        toxic_scores.append(hate_toxic)
    return toxic_scores
