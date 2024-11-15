import aiohttp
import asyncio
import evaluate
import json
import logging
import numpy as np
import sys

from googleapiclient import discovery
from os import getenv
from pathlib import Path
from transformers import pipeline
from typing import Optional, Dict, List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSPECTIVE_API_KEY = getenv("PERSPECTIVE_API_KEY", None)


def eval_on_metric(
    metric,
    continuations,
    ground_truths: Optional[List] = None,
    asynchronously: bool = True,
    batch_size: int = 8,
    noise=0,
):
    """
    Evaluate continuations on the specified metric.

    Args:
        metric (str): The evaluation metric to use.
        continuations (List[str]): The generated continuations to evaluate.
        ground_truths (Optional[List[str]]): The ground truth texts for translation metrics.
        asynchronously (bool): Whether to perform asynchronous evaluation (for 'perspective' metric).
        batch_size (int): The batch size to use for evaluation.

    Returns:
        List[float]: A list of scores corresponding to each continuation.

    """
    supported_metrics = {"perspective", "toxicity", "bleu", "rouge"}

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

    elif metric == "rouge" or metric == "bleu":
        if ground_truths is None:
            logger.error("Ground truths must be provided for translation evaluation.")
            sys.exit(1)

        scores = []
        # Load the metric
        score_function = evaluate.load_metric(metric)

        for mt, gt in zip(continuations, ground_truths):
            # Compute per-sample scores
            score = score_function.compute(
                predictions=[mt],
                references=[gt],
                use_aggregator=False,
                use_stemmer=True,
            )
            if metric == "bleu":
                scores.append(score["bleu"])
            else:
                scores.append(score["rougleLsum"])

    else:
        logger.error(f"Invalid metric provided. Supported metrics are: {', '.join(supported_metrics)}.")
        sys.exit(1)

    noisy_scores = [score + np.random.normal(0, noise) for score in scores]
    scores = np.clip(noisy_scores, 0, 1)

    return scores


def call_perspective_synchronously(continuations):
    """ """
    # from utils.keys import PERSPECTIVE_API_KEY

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


async def fetch_toxicity(session, text, retries=10):
    # from utils.keys import PERSPECTIVE_API_KEY

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
