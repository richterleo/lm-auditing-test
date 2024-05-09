# import importlib

from functools import partial

from deep_anytime_testing.data.datagen import DatasetOperator
from generate_and_evaluate import generate_and_evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader

import torch

from utils import time_block


class PromptDataset(DatasetOperator):
    def __init__(
        self,
        dataset_name,
        model_id1,
        model_kwargs1,
        gen_kwargs1,
        model_id2=None,
        model_kwargs2=None,
        gen_kwargs2=None,
        metric="toxicity",
        metric_kwargs=None,
    ):
        tau1 = partial(
            generate_and_evaluate,
            model_id1,
            model_kwargs1,
            gen_kwargs1,
            metric,
        )
        if model_id2 is None:
            tau2 = tau1
        else:
            tau2 = partial(
                generate_and_evaluate, model_id2, model_kwargs2, gen_kwargs2, metric
            )
        super().__init__(tau1, tau2)

        prompt_dataset = load_dataset(dataset_name, split="train")
        self.z = [sample["prompt"]["text"] for sample in prompt_dataset]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = "allenai/real-toxicity-prompts"
    prompt_dataset = PromptDataset(
        dataset_name,
        "meta-llama/Meta-Llama-3-8B",
        {
            "torch_dtype": torch.bfloat16,  # torch.float16
            "load_in_4bit": False,
            "device_map": "auto" if device == "cuda" else None,
            "attn_implementation": None,
        },
        {
            "max_length": 50,
            "do_sample": True,
            "temperature": 1,
        },
    )

    data_loader = DataLoader(prompt_dataset, batch_size=8, shuffle=True)

    data_iter = iter(data_loader)

    with time_block("Load Batch"):
        batch = next(data_iter)

    for sample in batch:
        print(sample)
