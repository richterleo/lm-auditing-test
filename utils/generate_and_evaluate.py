import evaluate
import json
import wandb
import aiohttp
import asyncio
import torch

from collections import defaultdict
from datasets import load_dataset
from googleapiclient import discovery
from tqdm import tqdm
from torch.utils.data import Subset
from transformers import pipeline, AutoTokenizer
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
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_and_evaluate(
    dataset_name: str,
    model_cfg,
    num_samples: int,
    batch_size: int = 8,
    seed=0,
    use_wandb=True,
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
        "batch_size": batch_size,
        "seed": seed,
        "use_wandb": use_wandb,
        "evaluate": evaluate,
        "metric": str(metric),
        "meta_data": meta_data,
    }

    if "Llama-3" in model_cfg["model_id"]:
        format_func = format_funcs["llama3"]
    elif "Mistral" in model_cfg["model_id"]:
        format_func = format_funcs["mistral"]
    elif "gemma" in model_cfg["model_id"]:
        format_func = format_funcs["gemma"]

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
        cont = (
            out[0]["generated_text"][len(prompt) :]
            .strip()
            .replace(prompt_dataset[i]["prompt"]["text"], "")
        )
        logs["prompts"].append(prompt_dataset[i]["prompt"]["text"])
        logs["continuations"].append(cont)

    file_name = f"{model_cfg['model_id'].split('/')[-1]}_continuations_seed{seed}.json"

    with open(file_name, "w") as file:
        json.dump(logs, file, indent=4)

    if use_wandb:
        wandb.save(file_name)


def eval_on_metric(metric, continuations, asynchronously=True, batch_size=8):
    """Evaluate the given metric on a list of continuations.

    Args:
        metric (str): The metric to evaluate. Possible values are "perspective" and "toxicity".
        continuations (list): A list of continuations to evaluate.
        asynchronously (bool, optional): Whether to evaluate asynchronously. Defaults to True.
        batch_size (int, optional): The batch size for toxicity evaluation. Defaults to 8.

    Returns:
        list: A list of scores corresponding to each continuation.

    Raises:
        ValueError: If an invalid metric is provided.

    """

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
            continuations,
            batch_size=batch_size,
        )

        scores = [score[1]["score"] for score in toxicity_scores]

    else:
        raise ValueError(
            "Invalid metric provided. Supported metrics are 'perspective' and 'toxicity'."
        )

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
        await asyncio.sleep(240)  # Wait a bit before retrying

    raise Exception(f"Failed to fetch toxicity data after {retries} attempts")


async def call_perspective(continuations):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_toxicity(session, text) for text in continuations]
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
