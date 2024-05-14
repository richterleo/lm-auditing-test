import importlib
import numpy as np
import logging
import random
import torch
import wandb

from copy import deepcopy
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from transformers import pipeline, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from tqdm import tqdm

# own utilities
from dah_testing.dataloader import ScoresDataset, collate_fn

# from arguments import Cfg
from utils.generate_and_evaluate import eval_on_metric
from utils.utils import translate_model_kwargs, time_block, NestedKeyDataset

deep_anytime_testing = importlib.import_module("deep-anytime-testing")
train = importlib.import_module("deep-anytime-testing.trainer.trainer")
Trainer = getattr(train, "Trainer")


class EvalTrainer(Trainer):
    def __init__(
        self,
        train_cfg,
        net,
        tau1_cfg,
        dataset_name,
        device,
        behavior,
        metric,
        use_wandb,
        tau2_cfg=None,
    ):
        use_same_tau = False
        if not tau2_cfg:
            tau2_cfg = tau1_cfg
            use_same_tau = True

        super().__init__(
            train_cfg, net, tau1_cfg, tau2_cfg, dataset_name, device, train_cfg.seed
        )

        self.tokenizer1 = AutoTokenizer.from_pretrained(
            tau1_cfg["model_id"], padding_side="left"
        )  # TODO: check whether the padding issue makes any difference

        if self.tokenizer1.pad_token is None:
            self.tokenizer1.pad_token_id = self.tokenizer1.eos_token_id

        model1_kwargs = translate_model_kwargs(tau1_cfg["model_kwargs"])
        if is_flash_attn_2_available():
            model1_kwargs.update({"attn_implementation": "flash_attention_2"})

        self.pipeline1 = pipeline(
            "text-generation",
            model=tau1_cfg["model_id"],
            # device=self.device,
            model_kwargs=model1_kwargs,
            tokenizer=self.tokenizer1,
            pad_token_id=self.tokenizer1.eos_token_id,
            device_map="auto",
        )
        # TODO: make this more efficient
        if use_same_tau:
            self.tokenizer2 = self.tokenizer1
            self.pipeline2 = self.pipeline1
        else:
            self.tokenizer2 = AutoTokenizer.from_pretrained(
                tau2_cfg["model_id"], padding_side="left"
            )
            if self.tokenizer2.pad_token is None:
                self.tokenizer2.pad_token_id = self.tokenizer2.eos_token_id

            model2_kwargs = translate_model_kwargs(tau2_cfg["model_kwargs"])
            if is_flash_attn_2_available():
                model2_kwargs.update({"attn_implementation": "flash_attention_2"})

            self.pipeline2 = pipeline(
                "text-generation",
                model=tau2_cfg["model_id"],
                # device=self.device,
                model_kwargs=model2_kwargs,
                tokenizer=self.tokenizer2,
                pad_token_id=self.tokenizer2.eos_token_id,
                device_map="auto",
            )

        # TODO: need to make sure that they're using different random seeds

        self.gen1_kwargs = tau1_cfg["gen_kwargs"]
        self.gen2_kwargs = (
            tau1_cfg["gen_kwargs"] if use_same_tau else tau2_cfg["gen_kwargs"]
        )

        self.behavior = behavior
        self.metric = metric if metric else behavior

        # we load in the whole dataset at once
        with time_block("Loading the dataset"):
            self.dataset = load_dataset(
                self.datagen, split="train"
            )  # TODO: Do we need the other mode as well?

        self.use_wandb = use_wandb

    def log(self, logs, seq, epoch, total_epoch, new_start_sequence):
        """
        Log metrics for visualization and monitoring.

        Args:
        - logs (dict): Dictionary containing metrics to be logged.
        """

        for key, value in logs.items():
            if self.use_wandb:
                wandb.log(
                    {
                        key: value,
                        "sequence": seq,
                        "epoch": epoch,
                        "epoch_total": total_epoch,
                        "new_start_sequence": new_start_sequence,
                    }
                )
            print(
                f"Seq: {self.current_seq}, Epoch: {self.current_epoch}, {key}: {value}"
            )

    def train(self):
        """
        Overwrite method from Trainer class
        """

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        davts = []

        self.current_seq = 0
        self.current_epoch = 0
        self.current_total_epoch = 0

        num_batches = (len(self.dataset) + self.bs - 1) // self.bs

        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        batches = [
            indices[i * self.bs : min((i + 1) * self.bs, len(self.dataset))]
            for i in range(num_batches)
        ]

        with time_block("Now evaluating on the first test_ds"):
            # in the first sequence, we don't train our model
            test_ds = self.get_score_ds(batches[0])
            test_loader = DataLoader(
                test_ds, batch_size=self.bs, shuffle=True, collate_fn=collate_fn
            )
            _, davt = self.train_evaluate_epoch(test_loader, mode="test")
            davts.append(davt.item())
            self.log(
                {"aggregated_test_e-value": davt},
                self.current_seq,
                self.current_epoch,
                self.current_total_epoch,
                int(self.current_epoch == 0),
            )

        # Log information if davt exceeds the threshold TODO: not sure we need this for first batch??
        if davt > (1.0 / self.alpha):
            logging.info("Reject null at %f", davt)
            self.log({"steps": 0}, self.current_seq, self.current_epoch, 0)

        for k in range(1, min(self.seqs, num_batches)):
            # This is the maximum number of mini-batches to sample from the data

            self.current_seq = k

            # If k=1, we still need to define train and val set
            if k == 1:
                # in this case, we need to define val set as fraction of train set
                batch_indices = batches[k - 1]
                train_indices, val_indices = train_test_split(
                    np.array(batch_indices), test_size=0.3, random_state=self.seed
                )
                # TODO: make this smoother
                train_indices = [ti.item() for ti in train_indices]
                val_indices = [vi.item() for vi in val_indices]
                train_ds = self.get_score_ds(train_indices)
                val_ds = self.get_score_ds(val_indices)
                train_loader = DataLoader(
                    train_ds, batch_size=self.bs, shuffle=True, collate_fn=collate_fn
                )
                val_loader = DataLoader(
                    val_ds, batch_size=self.bs, shuffle=True, collate_fn=collate_fn
                )

            # Actual model training
            for i in range(self.epochs):
                self.current_epoch = i
                self.current_total_epoch += 1
                with time_block(f"Training epoch {i} on sequence {k}"):
                    self.train_evaluate_epoch(train_loader)
                with time_block(f"Validation epoch {i} on sequence {k}"):
                    loss_val, _ = self.train_evaluate_epoch(val_loader, mode="val")

                # Check for early stopping or end of epochs
                if (
                    self.early_stopper.early_stop(loss_val.detach())
                    or (i + 1) == self.epochs
                ):
                    # Now define test data from current batch
                    batch_indices = batches[k]
                    test_ds = self.get_score_ds(batch_indices)
                    test_loader = DataLoader(
                        test_ds, batch_size=self.bs, shuffle=True, collate_fn=collate_fn
                    )

                    # Get S_t value on current batch
                    _, conditional_davt = self.train_evaluate_epoch(
                        test_loader, mode="test"
                    )
                    davts.append(conditional_davt.item())
                    davt = np.prod(np.array(davts[self.T :])) if k >= self.T else 1
                    self.log(
                        {"aggregated_test_e-value": davt},
                        self.current_seq,
                        self.current_epoch,
                        self.current_total_epoch,
                        int(self.current_epoch == 0),
                    )

                    # former train_ds and val_ds become the new train set
                    train_ds = ConcatDataset([train_ds, val_ds])
                    train_loader = DataLoader(
                        train_ds,
                        batch_size=self.bs,
                        shuffle=True,
                        collate_fn=collate_fn,
                    )

                    # former test_loader (i.e. current batch) becomes validation set
                    val_loader = test_loader

                    break

            # Reset the early stopper for the next sequence
            self.early_stopper.reset()

            # Log information if davt exceeds the threshold
            if davt > (1.0 / self.alpha):
                print("Reject null at %f", davt)
                self.log(
                    {"steps": k},
                    self.current_seq,
                    self.current_epoch,
                    self.current_total_epoch,
                    int(self.current_epoch == 0),
                )

    def train_evaluate_epoch(self, data_loader, mode="train"):
        """ """

        aggregated_loss = 0
        davt = 1
        num_samples = len(data_loader.dataset)

        self.log(
            {"num_samples": num_samples},
            self.current_seq,
            self.current_epoch,
            self.current_total_epoch,
            int(self.current_epoch == 0),
        )

        for batch in data_loader:
            if mode == "train":
                self.net.train()
                # values for tau1 and tau2
                tau1, tau2 = torch.split(batch, 1, dim=1)
                out = self.net(tau1, tau2)
            else:
                self.net.eval()
                tau1, tau2 = torch.split(batch, 1, dim=1)
                out = self.net(tau1, tau2).detach()

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
            },
            self.current_seq,
            self.current_epoch,
            self.current_total_epoch,
            int(self.current_epoch == 0),
        )
        return aggregated_loss / num_samples, davt

    def get_score_ds(
        self, indices, batch_size=8
    ):  # TODO: make this batch_size a param in configuration
        """
        Querying the models for continuations and evaluating them on the metric.
        """
        continuations1 = []
        continuations2 = []

        subset = Subset(self.dataset, indices)

        with time_block(f"Generating continuations for {len(indices)} samples"):
            # Get outputs from first pipeline
            for i, out in tqdm(
                enumerate(
                    self.pipeline1(
                        NestedKeyDataset(subset, "prompt", "text"),
                        pad_token_id=self.tokenizer1.eos_token_id,
                        batch_size=batch_size,
                        **self.gen1_kwargs,
                    )
                )
            ):
                cont1 = out[0]["generated_text"].replace(
                    subset[i]["prompt"]["text"], ""
                )
                continuations1.append(cont1)

            # Get outputs from second pipeline
            for i, out in tqdm(
                enumerate(
                    self.pipeline2(
                        NestedKeyDataset(subset, "prompt", "text"),
                        pad_token_id=self.tokenizer2.eos_token_id,
                        batch_size=batch_size,
                        **self.gen2_kwargs,
                    )
                )
            ):
                cont2 = out[0]["generated_text"].replace(
                    subset[i]["prompt"]["text"], ""
                )

                continuations2.append(cont2)

        # Get metrics for batch
        with time_block(f"Generating metric scores for {len(indices)} samples"):
            scores1 = eval_on_metric(self.metric, continuations1)
            scores2 = eval_on_metric(self.metric, continuations2)

        # Make new dataset
        score_ds = ScoresDataset(scores1, scores2)

        return score_ds

    def get_score_ds_slow(self, indices):  # TODO: remove this, this was pre-batching
        """
        Querying the models for continuations and evaluating them on the metric.
        """
        continuations1 = []
        continuations2 = []

        with time_block(f"Generating continuations for {len(indices)} samples"):
            for sample in list(indices):
                with time_block(
                    f"Generating continuation for sample {sample} out of {len(indices)}"
                ):
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
        with time_block(f"Generating metric scores for {len(indices)} samples"):
            scores1 = eval_on_metric(self.metric, continuations1)
            scores2 = eval_on_metric(self.metric, continuations2)

        # Make new dataset
        score_ds = ScoresDataset(scores1, scores2)

        return score_ds
