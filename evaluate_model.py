import json
import os
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, PreTrainedModel, PreTrainedTokenizer

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
    project=f"{args.metric}_evaluation", entity="richter-leo94", name=args.run_name
)

prompts = load_dataset(args.dataset_name, split="train")

pipeline = pipeline(
    "text-generation",
    model=args.model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "use_flash_attention_2": False,  # investigate why this does not work
        "load_in_4bit": False,
    },  # investigate why this does not work
    device=args.device,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_id)

sample = get_random_prompts(prompts)
prompts = [p["text"] for p in sample["prompt"]]

scores = {}

metric = evaluate.load(args.metric)  # Aggregate different metrics here

for epoch in tqdm(range(args.epochs)):
    model_continuations = []
    for prompt in tqdm(prompts):
        generation = pipeline(
            prompt, max_length=100, do_sample=True, temperature=2.0, pad_token_id=50256
        )

        continuation = generation[0]["generated_text"].replace(prompt, "")
        model_continuations.append(continuation)

    ratings = metric.compute(predictions=model_continuations)

    scores[epoch] = ratings[args.metric]
    print(scores[epoch])

    hist = np.histogram(scores[epoch], bins="auto")
    wandb.log({f"Epoch {epoch+1} Toxicity Scores": wandb.Histogram(np_histogram=hist)})

    log_scores(scores)

    file_name = f"{metric}_scores.json"

    with open(file_name, "w") as file:
        json.dump(scores, file, indent=4)

    print(f"Saved scores epoch {epoch} out of {args.epochs}.")

print("Finished evaluating.")

wandb.finish()
