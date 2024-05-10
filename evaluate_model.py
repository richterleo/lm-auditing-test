import evaluate
import json
import torch
import wandb

from copy import deepcopy
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from tqdm import tqdm

from arguments import EvalArgs
from utils import get_random_prompts, log_scores


device = "cuda" if torch.cuda.is_available() else "cpu"
args = EvalArgs(device=device, temperature=1.0, num_samples=15, epochs=3)

run = wandb.init(
    project=f"{args.metric}_evaluation",
    entity="richter-leo94",
    name=args.run_name,
    config={
        "dataset_name": args.dataset_name,
        "model_id": args.model_id,
        "metric": args.metric,
        "metric_lower_lim": args.lower_lim,
        "metric_upper_lim": args.upper_lim,
        "num_epochs": args.epochs,
        "num_samples": args.num_samples,
        "num_bins": args.num_bins,
        "device": device,
        "model_kwargs": {
            "torch_dtype": "torch.bfloat16",  # torch.float16
            "load_in_4bit": True,
            "device_map": "auto" if device == "cuda" else None,
            "attn_implementation": "flash_attention_2"
            if is_flash_attn_2_available()
            else None,
        },
        "generation_kwargs": {
            "max_length": args.max_length,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
        },
    },
)


prompt_dataset = load_dataset(args.dataset_name, split="train")

# wandb only logs strings, floats, ... so need to modify torch_dtype
model_kwargs = deepcopy(wandb.config.model_kwargs)
if model_kwargs["torch_dtype"] == "torch.bfloat16":
    model_kwargs["torch_dtype"] = torch.bfloat16
elif model_kwargs["torch_dtype"] == "torch.float16":
    model_kwargs["torch_dtype"] = torch.float16
else:
    model_kwargs["torch_dtype"] = "auto"


generation_kwargs = deepcopy(wandb.config.generation_kwargs)

generator = pipeline(
    "text-generation",
    model=args.model_id,
    # device=args.device, # no device param if using accelerate (load_in_4bit=True)
    model_kwargs=model_kwargs,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_id)

sample = get_random_prompts(prompt_dataset, num_examples=args.num_samples)
prompts = [p["text"] for p in sample["prompt"]]

results = {}

metric = evaluate.load(args.metric)  # Aggregate different metrics here

all_data_table = wandb.Table(columns=["epoch", "step", "ratings"])


for epoch in tqdm(range(args.epochs)):
    model_continuations = []
    for prompt in tqdm(prompts):
        generation = generator(
            prompt, pad_token_id=tokenizer.eos_token_id, **generation_kwargs
        )

        continuation = generation[0]["generated_text"].replace(prompt, "")
        model_continuations.append(continuation)

    ratings = metric.compute(predictions=model_continuations)

    results[epoch] = {}
    results[epoch]["generations"] = model_continuations
    results[epoch]["ratings"] = ratings[args.metric]

    for i, rating in enumerate(ratings[args.metric]):
        all_data_table.add_data(epoch, i, rating)

    # upload json to wandb
    log_scores(results)

    file_name = f"{args.metric}_scores.json"

    with open(file_name, "w") as file:
        json.dump(results, file, indent=4)

    print(f"Saved scores epoch {epoch} out of {args.epochs}.")


run.log(
    {
        "Ratings Histogram": wandb.plot.histogram(
            all_data_table,
            "ratings",
            title=f"{args.metric} ratings",
            # group_by="epoch"
        )
    }
)


print("Finished evaluating.")

wandb.finish()
