import evaluate
import json
import wandb
import aiohttp
import asyncio
import torch

from collections import defaultdict
from datasets import load_dataset
from googleapiclient import discovery
from utils.keys import PERSPECTIVE_API_KEY
from tqdm import tqdm
from torch.utils.data import Subset
from transformers import pipeline, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from peft import AutoPeftModelForCausalLM

from utils.utils import (
    translate_model_kwargs,
    get_random_prompts,
    log_scores,
    NestedKeyDataset,
    terminator,
    format_funcs,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_and_evaluate(
    dataset_name: str,
    model_cfg,
    num_samples: int,
    num_epochs: int = 1,
    batch_size: int = 8,
    save_continuations=True,  # TODO: add this flag
    save_prompts=True,  # TODO: add this flag
    seed=0,
    use_wandb=True,
    evaluate: bool = True,
    metric=None,
    meta_data=None,
):
    """ """

    prompt_dataset = load_dataset(dataset_name, split="train")

    # wandb only logs strings, floats, ... so need to modify torch_dtype
    model_kwargs = translate_model_kwargs(model_cfg["model_kwargs"])
    if is_flash_attn_2_available():
        model_kwargs.update({"attn_implementation": "flash_attention_2"})
    gen_kwargs = model_cfg["gen_kwargs"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_id"], padding_side="left"
    )

    terminators = [tokenizer.eos_token_id]

    if "Llama-3" in model_cfg["model_id"]:
        terminators.append(tokenizer.convert_tokens_to_ids(terminator["llama3"]))
    elif "Mistral" in model_cfg["model_id"]:
        terminators.append(tokenizer.convert_tokens_to_ids(terminator["mistral"]))
    elif "gemma" in model_cfg["model_id"]:
        terminators.append(tokenizer.convert_tokens_to_ids(terminator["gemma"]))

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_cfg["model_id"].startswith("LLMAccountability"):
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_cfg["model_id"], **model_kwargs
        )
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
            model=model_cfg["model_id"],
            model_kwargs=model_kwargs,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.pad_token_id,
        )

    torch.manual_seed(seed)

    if num_samples < len(prompt_dataset):
        subset_indices = torch.randperm(len(prompt_dataset))[:num_samples]
        prompt_dataset = Subset(prompt_dataset, subset_indices.tolist())

    logs = defaultdict(lambda: defaultdict(list))
    logs["metadata"] = {
        "dataset_name": dataset_name,
        "model_id": model_cfg["model_id"],
        "gen_kwargs": gen_kwargs,
        "num_samples": num_samples,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "seed": seed,
        "use_wandb": use_wandb,
        "evaluate": evaluate,
        "metric": str(metric),
        "meta_data": meta_data,
    }

    # For logging histogram
    if use_wandb:
        all_data_table = wandb.Table(columns=["epoch", "step", "ratings"])

    if "Llama-3" in model_cfg["model_id"]:
        format_func = format_funcs["llama3"]
    elif "Mistral" in model_cfg["model_id"]:
        format_func = format_funcs["mistral"]
    elif "gemma" in model_cfg["model_id"]:
        format_func = format_funcs["gemma"]

    # This loop is for repeated evaluation on the same prompts (default is only 1 epoch)
    for epoch in range(num_epochs):
        for i, out in tqdm(
            enumerate(
                generator(
                    NestedKeyDataset(
                        prompt_dataset,
                        "prompt",
                        "text",
                        model_cfg["model_id"],
                        tokenizer,
                    ),
                    batch_size=batch_size,
                    eos_token_id=terminators,
                    **gen_kwargs,
                )
            )
        ):
            prompt = tokenizer.apply_chat_template(
                format_func(prompt_dataset[i]["prompt"]["text"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "prompt": prompt_dataset[i]["prompt"]["text"],
                        "continuation": out[0]["generated_text"][len(prompt):].strip().replace(prompt_dataset[i]["prompt"]["text"], ""),
                    }
                )

            # cont = out[0]["generated_text"].replace(
            #    prompt_dataset[i]["prompt"]["text"], ""
            cont = out[0]["generated_text"][len(prompt):].strip().replace(prompt_dataset[i]["prompt"]["text"], "")
            logs[epoch]["prompts"].append(prompt_dataset[i]["prompt"]["text"])
            logs[epoch]["continuations"].append(cont)

        if evaluate:
            scores = eval_on_metric(metric, logs[epoch]["continuations"])
            logs[epoch][f"{str(metric)}_scores"] = scores

            if use_wandb:
                for i, score in enumerate(scores):
                    wandb.log({f"{str(metric)}_score": score, "samples": i})
                    all_data_table.add_data(epoch, i, score)

        file_name = (
            f"{str(metric)}_scores.json"
            if evaluate
            else f"{model_cfg['model_id'].split('/')[-1]}_continuations_seed{seed}.json"
        )

        with open(file_name, "w") as file:
            json.dump(logs, file, indent=4)

        if use_wandb:
            wandb.save(file_name)

    # plot histogram in wandb
    if use_wandb and evaluate:
        wandb.log(
            {
                "Ratings Histogram": wandb.plot.histogram(
                    all_data_table,
                    "ratings",
                    title=f"{str(metric)}_scores",
                )
            }
        )


def eval_on_metric(metric, continuations):
    """ """

    if metric == "perspective":
        ratings = asyncio.run(call_perspective(continuations))

    else:
        metric_name = metric
        metric = evaluate.load(metric)
        rating_dict = metric.compute(predictions=continuations)
        ratings = rating_dict[metric_name]

    return ratings


# def call_perspective(continuations):
#     """ """
#     client = discovery.build(
#         "commentanalyzer",
#         "v1alpha1",
#         developerKey=PERSPECTIVE_API_KEY,
#         discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
#         static_discovery=False,
#     )

#     ratings = []

#     for continuation in continuations:
#         analyze_request = {
#             "comment": {"text": continuation},
#             "requestedAttributes": {"TOXICITY": {}},
#             "doNotStore": True,
#         }

#     response = client.comments().analyze(body=analyze_request).execute()
#     ratings.append(response["attributeScores"]["TOXICITY"]["summaryScore"]["value"])

#     return ratings


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


async def fetch_toxicity(session, text, retries=3):
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    params = {
        "key": PERSPECTIVE_API_KEY,
    }
    payload = {
        "comment": {"text": text},
        "requestedAttributes": {"TOXICITY": {}},
        "doNotStore": True,
    }
    headers = {"Content-Type": "application/json"}

    for attempt in range(retries):
        try:
            async with session.post(
                url,
                params=params,
                data=json.dumps(payload),
                headers=headers,
                timeout=10,
            ) as response:
                if response.status == 200:
                    resp_json = await response.json()
                    return resp_json["attributeScores"]["TOXICITY"]["summaryScore"][
                        "value"
                    ]
                else:
                    logger.warning(
                        f"Attempt {attempt + 1}: Received status code {response.status}"
                    )
                    logger.warning(f"Response content: {await response.text()}")
        except aiohttp.ClientError as e:
            logger.error(f"Attempt {attempt + 1}: ClientError - {e}")
        except asyncio.TimeoutError:
            logger.error(f"Attempt {attempt + 1}: Request timed out")
        await asyncio.sleep(1)  # Wait a bit before retrying

    raise Exception(f"Failed to fetch toxicity data after {retries} attempts")


async def call_perspective(continuations):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_toxicity(session, text) for text in continuations]
        ratings = await asyncio.gather(*tasks)
    return ratings
