import json
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
        "metric_lower_lim": args.lower_lim,
        "metric_upper_lim": args.upper_lim,
        "num_epochs": args.epochs,
        "num_samples": args.num_samples,
        "num_bins": args.num_bins,
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,  # torch.float16,  # TODO: find out why torch.bfloat16 does not work
            "load_in_4bit": True,
            "device_map": "auto",
            # Add flash_attention_2 if available
            **(
                {"attn_implementation": "flash_attention_2"}
                if is_flash_attn_2_available()
                else {}
            ),
        },
        "generation_kwargs": {"max_length": 50, "do_sample": True, "temperature": 2.0},
    },
)


prompt_dataset = load_dataset(args.dataset_name, split="train")

# wandb only logs strings, floats, ... so need to modify torch_dtype
model_kwargs = wandb.config.model_kwargs
if model_kwargs["torch_dtype"] == "torch.bfloat16":
    model_kwargs["torch_dtype"] = torch.bfloat16
elif model_kwargs["torch_dtype"] == "torch.float16":
    model_kwargs["torch_dtype"] = torch.float16
else:
    model_kwargs["torch_dtype"] = "auto"

generator = pipeline(
    "text-generation",
    model=args.model_id,
    # device=args.device, # no device param if using accelerate
    model_kwargs=model_kwargs,
)

model_device = generator.device
print(f"The model is on device: {model_device}")

tokenizer = AutoTokenizer.from_pretrained(args.model_id)

sample = get_random_prompts(prompt_dataset, num_examples=args.num_samples)
prompts = [p["text"] for p in sample["prompt"]]

results = {}

metric = evaluate.load(args.metric)  # Aggregate different metrics here

wandb.config.generation_kwargs

for epoch in tqdm(range(args.epochs)):
    model_continuations = []
    for prompt in tqdm(prompts):
        generation = generator(
            prompt,
            pad_token_id=tokenizer.eos_token_id,
            generation_kwargs=wandb.config.generation_kwargs,
        )

        continuation = generation[0]["generated_text"].replace(prompt, "")
        model_continuations.append(continuation)

    ratings = metric.compute(predictions=model_continuations)

    results[epoch] = {}
    # results[epoch]["generations"] = model_continuations
    results[epoch]["ratings"] = ratings[args.metric]

    # hist_values, bin_edges = np.histogram(
    #     results[epoch]["ratings"],
    #     bins=args.num_bins,
    #     lower_lim=args.lower_lim,
    #     upper_lim=args.lower_lim,
    # )

    # # Convert histogram to a format that can be logged by wandb
    # wandb_hist_data = [[x, y] for x, y in zip(bin_edges[:-1], hist_values)]

    # # Log the histogram to wandb
    # wandb.log(
    #     {
    #         f"Epoch_{epoch}_histogram": wandb.plot.histogram(
    #             wandb_hist_data,
    #             value=f"{args.metric}_ratings",
    #             title=f"{args.metric}_ratings Histogram Epoch {epoch}",
    #         )
    #     }
    # )

    # upload json to wandb
    log_scores(results)

    file_name = f"{args.metric}_scores.json"

    with open(file_name, "w") as file:
        json.dump(results, file, indent=4)

    print(f"Saved scores epoch {epoch} out of {args.epochs}.")

print("Finished evaluating.")

wandb.finish()
