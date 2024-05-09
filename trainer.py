import hydra
import numpy as np
import logging
import random
import torch

from sklearn.model_selection import train_test_split
from datasets import load_dataset
from torch.utils.data import DataLoader, ConcatDataset
from transformers import pipeline, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from transformers.pipelines.pt_utils import KeyDataset

from hydra.utils import instantiate

from torch.utils.data import Dataset


from dataloader import ScoresDataset
from deep_anytime_testing.trainer.trainer import Trainer
from deep_anytime_testing.models.mlp import MMDEMLP

from arguments import Cfg

from generate_and_evaluate import eval_on_metric


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

        # we load in the whole dataset at once
        self.dataset = load_dataset(
            self.datagen, split="train"
        )  # TODO: Do we need the other mode as well?

    def train(self):
        """
        Overwrite method from Trainer class
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        davts = []

        num_batches = (len(self.dataset) + self.bs - 1) // self.bs

        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        batches = [
            indices[i * self.bs : min((i + 1) * self.bs, len(self.dataset))]
            for i in range(num_batches)
        ]

        # in the first sequence, we don't train our model
        test_ds = self.get_score_ds(batches[0])
        test_loader = DataLoader(test_ds, batch_size=self.bs, shuffle=True)
        _, davt = self.train_evaluate_epoch(test_loader, mode="test")
        davts.append(davt.item())
        self.log({"aggregated_test_e-value": davt})

        # Log information if davt exceeds the threshold TODO: not sure we need this for first batch??
        if davt > (1.0 / self.alpha):
            logging.info("Reject null at %f", davt)
            self.log({"steps": 0})

        for k in range(1, np.min(self.seqs, num_batches)):
            # This is the maximum number of mini-batches to sample from the data

            self.current_seq = k

            # If k=1, we still need to define train and val set
            if k == 1:
                # in this case, we need to define val set as fraction of train set
                batch_indices = batches[k - 1]
                train_indices, val_indices = train_test_split(
                    np.arrays(batch_indices), test_size=0.3, random_state=self.seed
                )
                train_ds = self.get_score_ds(train_indices)
                val_ds = self.get_score_ds(val_indices)
                train_loader = DataLoader(train_ds, batch_size=self.bs, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=self.bs, shuffle=True)

            # Actual model training
            for i in range(self.epochs):
                self.current_epoch = 1
                self.train_evaluate_epoch(train_loader)
                loss_val, _ = self.train_evaluate_epoch(val_loader, mode="val")

                # Check for early stopping or end of epochs
                if (
                    self.early_stopper.early_stop(loss_val.detach())
                    or (i + 1) == self.epochs
                ):
                    # Now define test data from current batch
                    batch_indices = batches[k]
                    test_ds = self.get_score_ds(batch_indices)
                    test_loader = DataLoader(test_ds, batch_size=self.bs, shuffle=True)

                    # Get S_t value on current batch
                    _, conditional_davt = self.train_evaluate_epoch(
                        test_loader, mode="test"
                    )
                    davts.append(conditional_davt.item())
                    davt = np.prod(np.array(davts[self.T :])) if k >= self.T else 1
                    self.log({"aggregated_test_e-value": davt})

                    # former train_ds and val_ds become the new train set
                    train_ds = ConcatDataset([train_ds, val_ds])
                    train_loader = DataLoader(
                        train_ds, batch_size=self.bs, shuffle=True
                    )

                    # former test_loader (i.e. current batch) becomes validation set
                    val_loader = test_loader

                    break

            # Reset the early stopper for the next sequence
            self.early_stopper.reset()

            # Log information if davt exceeds the threshold
            if davt > (1.0 / self.alpha):
                print("Reject null at %f", davt)
                self.log({"steps": k})

    def train_evaluate_epoch(self, data_loader, mode="train"):
        """ """

        aggregated_loss = 0
        davt = 1
        num_samples = len(data_loader.dataset)

        for score1, score2 in data_loader:
            if mode == "train":
                self.net.train()
                out = self.net(score1, score2)
            else:
                self.net.eval()
                out = self.net(score1, score2).detach()

            loss = -out.mean() + self.l1_lambda * self.l1_regularization()
            aggregated_loss += -out.sum()
            davt *= torch.exp(out.sum())
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.log(
            {
                f"{mode}_e-value": davt.item(),
                f"{mode}_loss": aggregated_loss.item() / num_samples,
            }
        )
        return aggregated_loss / num_samples, davt

    def get_score_ds(self, indices):
        """ """
        scores = []
        continuations1 = []
        continuations2 = []

        for sample in list(indices):
            out1 = self.pipeline1(
                self.dataset[sample]["prompt"]["text"],
                pad_token_id=self.tokenizer1.eos_token_id,
                **self.gen1_kwargs,
            )
            out2 = self.pipeline2(
                self.dataset[sample]["prompt"]["text"],
                pad_token_id=self.tokenizer2.eos_token_id,
                **self.gen2_kwargs,
            )

            cont1 = out1[0]["generated_text"].replace(
                self.dataset[sample]["prompt"]["text"], ""
            )
            cont2 = out2[0]["generated_text"].replace(
                self.dataset[sample]["prompt"]["text"], ""
            )

            continuations1.append(cont1)
            continuations2.append(cont2)

        # Get metrics for batch
        score1 = eval_on_metric(self.metric, continuations1)
        score2 = eval_on_metric(self.metric, continuations2)

        scores.append((score1, score2))

        # Make new dataset
        score_ds = ScoresDataset(scores)

        return score_ds


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
    trainer.train()
