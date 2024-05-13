import evaluate
import json
import wandb

from collections import defaultdict
from datasets import load_dataset
from googleapiclient import discovery
from utils.keys import PERSPECTIVE_API_KEY
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

from utils.utils import translate_model_kwargs, get_random_prompts, log_scores


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
        logs = defaultdict(list)

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
        wandb.log(
            {
                f"{metric}_scores/{metric}_score-{i}": score
                for i, score in enumerate(scores)
            }
        )

        if comp_model_cfg:
            comp_scores = eval_on_metric(metric, logs[epoch]["comp_continuatiosn"])
            logs[epoch][f"{metric}_comp_scores"] = comp_scores
            wandb.log(
                {
                    f"{metric}_comp_scores/{metric}_comp_score-{i}": score
                    for i, score in enumerate(comp_scores)
                }
            )

        for i, score in enumerate(scores["epoch"][f"{metric}_scores"]):
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
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=PERSPECTIVE_API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

        analyze_request = {
            "comment": {"text": continuations[0]},
            "requestedAttributes": {"TOXICITY": {}},
            "doNotStore": True,
        }

        response = client.comments().analyze(body=analyze_request).execute()
        ratings = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]

    else:
        metric_name = metric
        metric = evaluate.load(metric)
        rating_dict = metric.compute(predictions=continuations)
        ratings = rating_dict[metric_name]
        print(type(ratings), ratings.shape)

    return ratings


if __name__ == "__main__":
    # continuations = ["This is great!", "I hate this!"]
    # metric = "perspective"

    # ratings = eval_on_metric(metric, continuations)
    # print(ratings)
    pass
