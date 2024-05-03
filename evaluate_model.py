import json
import numpy as np

from transformers import pipeline, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

from datasets import load_dataset
import evaluate
import torch
from tqdm import tqdm
import wandb

from arguments import EvalArgs
from utils import get_random_prompts, log_scores

device = "cuda" if torch.cuda.is_available() else "cpu"
args = EvalArgs(device=device)

wandb.init(
    project=f"{args.metric}_evaluation",
    entity="richter-leo94",
    name=args.run_name,
    config={
        "dataset_name": args.dataset_name,
        "model_id": args.model_id,
        "metric": args.metric,
        "num_epochs": args.epochs,
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "load_in_4bit": False,
            # Add flash_attention_2 if available
            **(
                {"attn_implementation": "flash_attention_2"}
                if is_flash_attn_2_available()
                else {}
            ),
        },
        "generation_kwargs": {
            "max_length": 100,
            "do_sample": True,
            "temperature": 2.0,
            "pad_token_id": 50256,
            "batch_size": 8,
        },
    },
)


prompts = load_dataset(args.dataset_name, split="train")

pipeline = pipeline(
    "text-generation",
    model=args.model_id,
    model_kwargs=wandb.config.model_kwargs,
    device=args.device,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_id)

sample = get_random_prompts(prompts)
prompts = [p["text"] for p in sample["prompt"]]

results = {}

metric = evaluate.load(args.metric)  # Aggregate different metrics here


for epoch in tqdm(range(args.epochs)):
    model_continuations = []
    for prompt in tqdm(prompts):
        generation = pipeline(prompt, generation_kwargs=wandb.config.generation_kwargs)

        continuation = generation[0]["generated_text"].replace(prompt, "")
        model_continuations.append(continuation)

    ratings = metric.compute(predictions=model_continuations)

    results[epoch]["generation"] = model_continuations
    results[epoch]["rating"] = ratings[args.metric]

    hist = np.histogram(results[epoch], bins="auto")
    wandb.log(
        {f"Epoch {epoch+1} {args.metric} scores": wandb.Histogram(np_histogram=hist)}
    )

    log_scores(results)

    file_name = f"{args.metric}_scores.json"

    with open(file_name, "w") as file:
        json.dump(results, file, indent=4)

    print(f"Saved scores epoch {epoch} out of {args.epochs}.")

print("Finished evaluating.")

wandb.finish()
