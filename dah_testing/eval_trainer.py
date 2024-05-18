import importlib
import numpy as np
import logging
import random
import torch
import wandb

from copy import deepcopy
from sklearn.model_selection import train_test_split, KFold
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from transformers import pipeline, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from tqdm import tqdm

# own utilities
from dah_testing.dataloader import ScoresDataset, collate_fn, load_into_scores_ds

# from arguments import Cfg
from utils.generate_and_evaluate import eval_on_metric
from utils.utils import translate_model_kwargs, time_block, NestedKeyDataset, terminator

deep_anytime_testing = importlib.import_module("deep-anytime-testing")
train = importlib.import_module("deep-anytime-testing.trainer.trainer")
Trainer = getattr(train, "Trainer")


class OnlineTrainer(Trainer):
    def __init__(
        self,
        train_cfg,
        net,
        tau1_cfg,
        dataset_name,
        behavior,
        metric,
        use_wandb,
        tau2_cfg=None,
    ):
        super().__init__(
            train_cfg,
            net,
            tau1_cfg,
            tau2_cfg or tau1_cfg,
            dataset_name,
            None,
            train_cfg.seed,
        )

        self.use_same_tau = tau2_cfg is None

        self.pipeline1, self.tokenizer1, self.terminators1 = self.setup_model(self.tau1)
        self.pipeline2, self.tokenizer2, self.terminators2 = (
            (self.pipeline1, self.tokenizer1, self.terminators1)
            if self.use_same_tau
            else self.setup_model(self.tau2)
        )

        self.gen1_kwargs = self.tau1["gen_kwargs"]
        self.gen2_kwargs = (
            self.tau1["gen_kwargs"] if self.use_same_tau else self.tau2["gen_kwargs"]
        )

        self.behavior = behavior
        self.metric = metric if metric else behavior

        # Load the dataset
        with time_block("Loading the dataset"):
            self.dataset = load_dataset(self.datagen, split="train")

        self.use_wandb = use_wandb

    def setup_model(self, tau_cfg):
        """ """
        tokenizer = AutoTokenizer.from_pretrained(
            tau_cfg["model_id"], padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs = translate_model_kwargs(tau_cfg["model_kwargs"])
        if is_flash_attn_2_available():
            model_kwargs.update({"attn_implementation": "flash_attention_2"})

        terminators = [tokenizer.eos_token_id]
        terminator_key = self.get_terminator_key(tau_cfg["model_id"])
        if terminator_key:
            terminators.append(
                tokenizer.convert_tokens_to_ids(terminator[terminator_key])
            )

        model = (
            AutoPeftModelForCausalLM.from_pretrained(
                tau_cfg["model_id"], **model_kwargs
            )
            if tau_cfg["model_id"].startswith("LLMAccountability")
            else tau_cfg["model_id"]
        )

        pipeline_obj = pipeline(
            "text-generation",
            model=model,
            model_kwargs=model_kwargs,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
        )

        return pipeline_obj, tokenizer, terminators

    def get_terminator_key(self, model_id):
        """ """
        if "Llama-3" in model_id:
            return "llama3"
        elif "Mistral" in model_id:
            return "mistral"
        elif "gemma" in model_id:
            return "gemma"
        return None

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
        self, indices
    ):  # TODO: make this batch_size a param in configuration
        """
        Querying the models for continuations and evaluating them on the metric.
        """
        continuations1 = []
        continuations2 = []

        subset = Subset(self.dataset, indices)

        print(len(indices))

        with time_block(f"Generating continuations for {len(indices)} samples"):
            # Get outputs from first pipeline
            for out in tqdm(
                self.pipeline1(
                    NestedKeyDataset(
                        subset,
                        "prompt",
                        "text",
                        self.tau1["model_id"],
                        self.tokenizer1,
                    ),
                    pad_token_id=self.tokenizer1.eos_token_id,
                    batch_size=self.tau1["gen_batch_size"],
                    **self.gen1_kwargs,
                )
            ):
                cont1 = out[0]["generated_text"]
                continuations1.append(cont1)

            # Get outputs from second pipeline
            for out in tqdm(
                self.pipeline2(
                    NestedKeyDataset(
                        subset,
                        "prompt",
                        "text",
                        self.tau2["model_id"],
                        self.tokenizer2,
                    ),
                    pad_token_id=self.tokenizer2.eos_token_id,
                    batch_size=self.tau2["gen_batch_size"],
                    **self.gen2_kwargs,
                )
            ):
                cont2 = out[0]["generated_text"]
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


class OfflineTrainer(Trainer):
    def __init__(
        self,
        train_cfg,
        net,
        model_name1,
        seed1,
        model_name2,
        seed2,
        metric="toxicity",
        use_wandb=True,
        fold_num=None,
    ):
        super().__init__(
            train_cfg,
            net,
            (model_name1, seed1),
            (model_name2, seed2),
            None,
            None,
            train_cfg.seed,
        )

        # remove unnecessary attributes
        del self.datagen
        del self.device

        self.fold_num = fold_num
        print(f"Fold number: {self.fold_num}")
        self.metric = metric

        self.dataset = load_into_scores_ds(
            model_name1, seed1, model_name2, seed2, metric, fold_num=fold_num
        )
        self.num_batches = (len(self.dataset) + self.bs - 1) // self.bs
        self.batches = self.get_kfold_batches()

        self.use_wandb = use_wandb

        self.current_total_epoch = 0

    def log(self, logs, seq, epoch, total_epoch, new_start_sequence):
        """
        Log metrics for visualization and monitoring.

        Args:
        - logs (dict): Dictionary containing metrics to be logged.
        """

        for key, value in logs.items():
            if self.use_wandb:
                if self.fold_num:
                    wandb.log(
                        {
                            key: value,
                            "sequence": seq,
                            "epoch": epoch,
                            "epoch_total": total_epoch,
                            "new_start_sequence": new_start_sequence,
                            "fold_num": self.fold_num,
                        }
                    )
                else:
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

    def get_kfold_batches(self):
        kf = KFold(n_splits=self.num_batches, shuffle=True, random_state=self.seed)

        batches = []
        for i, (_, batch_indices) in enumerate(kf.split(self.dataset)):
            batches.append(Subset(self.dataset, batch_indices))

        return batches

    def train(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        davts = []

        self.current_seq = 0
        self.current_epoch = 0
        self.current_total_epoch = 0

        # In the first sequence, we don't train our model, directly evaluate
        test_ds = self.batches[0]
        self.num_samples = len(test_ds)
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

        # In first sequence, we need to distribute the data into train and val set
        train_ds, val_ds = train_test_split(
            self.batches[0], test_size=0.3, random_state=self.seed
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.bs, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.bs, shuffle=True, collate_fn=collate_fn
        )

        for k in range(1, min(self.seqs, self.num_batches)):
            self.current_seq = k
            self.current_epoch = 0

            with time_block(f"Sequence {k}"):
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
                        # Now define new test data from current batch
                        test_ds = self.batches[k]
                        print(
                            f"This is the length of the {k}-th test dataset: {len(test_ds)}"
                        )
                        print(
                            f"self.num_samples is {self.num_samples} BEFORE adding new test dataset batch"
                        )
                        self.num_samples += len(test_ds)
                        test_loader = DataLoader(
                            test_ds,
                            batch_size=self.bs,
                            shuffle=True,
                            collate_fn=collate_fn,
                        )
                        print(
                            f"self.num_samples is {self.num_samples} AFTER adding new test dataset batch"
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
                        print(
                            f"This is the current length of the train ds at sequence {k}: {len(train_ds)}"
                        )
                        train_loader = DataLoader(
                            train_ds,
                            batch_size=self.bs,
                            shuffle=True,
                            collate_fn=collate_fn,
                        )

                        # former test_loader (i.e. current batch) becomes validation set
                        val_ds = test_ds
                        val_loader = test_loader

                        break

            # Reset the early stopper for the next sequence
            self.early_stopper.reset()

            # Log information if davt exceeds the threshold
            if davt > (1.0 / self.alpha):
                print("Reject null at %f", davt)
                self.log(
                    {"steps": k, "total_num_samples": self.num_samples},
                    self.current_seq,
                    self.current_epoch,
                    self.current_total_epoch,
                    int(self.current_epoch == 0),
                )

                break

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
