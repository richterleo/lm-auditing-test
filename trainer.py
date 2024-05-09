import hydra
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader, ConcatDataset
from transformers import pipeline, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from transformers.pipelines.pt_utils import KeyDataset

from hydra.utils import instantiate

from torch.utils.data import Dataset


from deep_anytime_testing.trainer.trainer import Trainer
from deep_anytime_testing.models.mlp import MMDEMLP

from arguments import Cfg

from generate_and_evaluate import get_score


class EvalTrainer(Trainer):
    def __init__(
        self,
        cfg,
        net,
        tau1_cfg,
        dataset_name,
        device,
        data_seed,
        metric="toxicity",
        tau2_cfg=None,
    ):
        if tau2_cfg is None:
            use_same_tau = True
            tau2_cfg = tau1_cfg
        super().__init__(cfg, net, tau1_cfg, tau2_cfg, dataset_name, device, data_seed)

        self.tokenizer1 = AutoTokenizer.from_pretrained(tau1_cfg["model_id"])

        if self.tokenizer1.pad_token is None:
            self.tokenizer1.pad_token_id = self.tokenizer1.eos_token_id
        self.pipeline1 = pipeline(
            "text-generation",
            model=tau1_cfg["model_id"],
            # device=self.device,
            model_kwargs=tau1_cfg["model_kwargs"],
            tokenizer=self.tokenizer1,
            pad_token_id=self.tokenizer1.eos_token_id,
        )

        if use_same_tau:
            self.tokenizer2 = self.tokenizer1
            self.pipeline2 = self.pipeline1
        else:
            self.tokenizer2 = AutoTokenizer.from_pretrained(tau2_cfg["model_id"])
            if self.tokenizer2.pad_token is None:
                self.tokenizer2.pad_token_id = self.tokenizer2.eos_token_id

            self.pipeline2 = pipeline(
                "text-generation",
                model=tau2_cfg["model_id"],
                # device=self.device,
                model_kwargs=tau2_cfg["model_kwargs"],
                tokenizer=self.tokenizer2,
                pad_token_id=self.tokenizer2.eos_token_id,
            )

        # TODO: need to make sure that they're using different random seeds

        self.gen1_kwargs = tau1_cfg["gen_kwargs"]
        self.gen2_kwargs = (
            tau1_cfg["gen_kwargs"] if use_same_tau else tau2_cfg["gen_kwargs"]
        )

        self.metric = metric

    def load_data(self, seed, mode="train"):
        """
        Overwrite method from Trainer class.
        We load just the prompt dataset, apply tau later
        """
        # TODO: put random seed here
        dataset = load_dataset(self.datagen, split=mode)
        # dataloader = DataLoader(dataset, batch_size=self.bs, shuffle=True)

        return dataset

    def train_evaluate_epoch(self, dataset, mode="train", batching=True):
        """
        Overwrite method from Trainer class
        """
        # aggregated_loss = 0
        # davt = 1
        # num_samples = len(dataset)

        if batching:
            for i, (out1, out2) in enumerate(
                zip(
                    self.pipeline1(
                        NestedKeyDataset(dataset, "prompt", "text"),
                        batch_size=self.bs,
                        pad_token_id=self.tokenizer1.eos_token_id,
                        truncation="only_first",
                        **self.gen1_kwargs,
                    ),
                    self.pipeline2(
                        NestedKeyDataset(dataset, "prompt", "text"),
                        batch_size=self.bs,
                        pad_token_id=self.tokenizer2.eos_token_id,
                        truncation="only_first",
                        **self.gen2_kwargs,
                    ),
                )
            ):
                res1 = get_score(self.metric, out1[0]["generated_text"])
                res2 = get_score(self.metric, out2[0]["generated_text"])
                print(f"This is episode {i} and score 1 is {res1}, score 2 is {res2}")

        else:
            for i, sample in enumerate(dataset):
                out1 = self.pipeline1(
                    sample["prompt"]["text"],
                    pad_token_id=self.tokenizer1.eos_token_id,
                    **self.gen1_kwargs,
                )
                out2 = self.pipeline2(
                    sample["prompt"]["text"],
                    pad_token_id=self.tokenizer2.eos_token_id,
                    **self.gen2_kwargs,
                )

                continuation1 = out1[0]["generated_text"].replace(
                    sample["prompt"]["text"], ""
                )
                continuation2 = out2[0]["generated_text"].replace(
                    sample["prompt"]["text"], ""
                )
                res1 = get_score(self.metric, continuation1)
                res2 = get_score(self.metric, continuation2)
                print(
                    f"This is sample {i} out of {len(dataset)} and score 1 is {res1}, score 2 is {res2}"
                )
                print(
                    f"This is the first continuation: {out1[0]['generated_text']} and this is the second: {out2[0]['generated_text']}"
                )


class NestedKeyDataset(Dataset):
    def __init__(self, dataset: Dataset, key1: str, key2: str):
        self.dataset = dataset
        self.key1 = key1
        self.key2 = key2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i][self.key1][self.key2]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = "allenai/real-toxicity-prompts"

    net = MMDEMLP(6, [10, 10, 10], 1, True, False, 0.4, False)

    config = Cfg()
    tau1_cfg = {
        "model_id": "meta-llama/Meta-Llama-3-8B",
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,  # torch.float16
            "load_in_4bit": False,
            "device_map": "auto" if device == "cuda" else None,
            "attn_implementation": None,
        },
        "gen_kwargs": {"max_length": 50, "do_sample": True, "temperature": 1},
    }

    trainer = EvalTrainer(config, net, tau1_cfg, dataset_name, device, 0)
    dataset = trainer.load_data(0)
    trainer.train_evaluate_epoch(dataset)
