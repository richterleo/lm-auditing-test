import evaluate
import json
import wandb
import aiohttp
import asyncio

from collections import defaultdict
from datasets import load_dataset
from googleapiclient import discovery
from utils.keys import PERSPECTIVE_API_KEY
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

from utils.utils import translate_model_kwargs, get_random_prompts, log_scores

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_and_evaluate(
    dataset_name: str,
    metric: str,
    model_cfg,
    num_samples,
    num_epochs: int = 1,
    comp_model_cfg=None,
    save_continuations=True,
    save_prompts=False,
    seed=0,
    use_wandb=True,
):
    """ """

    prompt_dataset = load_dataset(dataset_name, split="train")

    # wandb only logs strings, floats, ... so need to modify torch_dtype
    model_kwargs = translate_model_kwargs(model_cfg["model_kwargs"])
    gen_kwargs = model_cfg["gen_kwargs"]

    generator = pipeline(
        "text-generation",
        model=model_cfg["model_id"],
        # device=args.device, # no device param if using accelerate (load_in_4bit=True)
        model_kwargs=model_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])

    # if you wand to get model comparison on the same prompts
    if comp_model_cfg:
        comp_model_kwargs = translate_model_kwargs(comp_model_cfg["model_kwargs"])
        comp_gen_kwargs = comp_model_cfg["gen_kwargs"]

        comp_generator = pipeline(
            "text-generation",
            model=comp_model_cfg["model_id"],
            # device=args.device, # no device param if using accelerate (load_in_4bit=True)
            model_kwargs=comp_model_kwargs,
        )

    sample = get_random_prompts(prompt_dataset, num_examples=num_samples)
    prompts = [p["text"] for p in sample["prompt"]]

    results = {}

    all_data_table = wandb.Table(columns=["epoch", "step", "ratings"])

    for epoch in range(num_epochs):
        logs = defaultdict(lambda: defaultdict(list))

        for prompt in tqdm(prompts):
            generation = generator(
                prompt, pad_token_id=tokenizer.eos_token_id, **gen_kwargs
            )

            continuation = generation[0]["generated_text"].replace(prompt, "")
            logs[epoch]["prompts"].append(prompt)
            logs[epoch]["continuations"].append(continuation)
            wandb.log({"prompt": prompt, "continuation": continuation})

            if comp_model_cfg:
                comp_generation = comp_generator(
                    prompt, pad_token_id=tokenizer.eos_token_id, **comp_gen_kwargs
                )

                comp_continuation = comp_generation[0]["generated_text"].replace(
                    prompt, ""
                )
                logs[epoch]["comp_continuations"].append(comp_continuation)
                wandb.log({"comp_continuation": comp_continuation})

        scores = eval_on_metric(metric, logs[epoch]["continuations"])
        logs[epoch][f"{metric}_scores"] = scores

        for i, score in enumerate(scores):
            wandb.log({f"{metric}_score": score, "samples": i})

        if comp_model_cfg:
            comp_scores = eval_on_metric(metric, logs[epoch]["comp_continuation"])
            logs[epoch][f"{metric}_comp_scores"] = comp_scores
            for i, score in enumerate(comp_scores):
                wandb.log({f"{metric}_comp_score": score, "samples": i})

        for i, score in enumerate(scores):
            all_data_table.add_data(epoch, i, score)

        # upload json to wandb
        log_scores(logs)

        file_name = f"{metric}_scores.json"

        with open(file_name, "w") as file:
            json.dump(results, file, indent=4)

        print(f"Saved scores epoch {epoch} out of {num_epochs}.")

        return logs, all_data_table


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
