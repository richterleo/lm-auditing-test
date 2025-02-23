import logging
import numpy as np
import pandas as pd
import random
import sys
import time
import torch
import wandb
import warnings

from datasets import load_dataset
from pathlib import Path
from peft import AutoPeftModelForCausalLM
from scipy.stats import wasserstein_distance, ks_2samp, ttest_ind, anderson_ksamp
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from transformers import pipeline, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from tqdm import tqdm
from typing import Optional, Dict, List

# own utilities
from lm_auditing.auditing.dataloader import ScoresDataset, collate_fn, load_into_scores_ds
from lm_auditing.evaluation.score import eval_on_metric
from lm_auditing.utils.utils import translate_model_kwargs, time_block, NestedKeyDataset, terminator

from lm_auditing.utils.dat_wrapper import Trainer, EarlyStopper

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class OfflineTrainer(Trainer):
    def __init__(
        self,
        train_cfg,
        net,
        model_name1,
        seed1,
        model_name2,
        seed2,
        metric="perspective",
        use_wandb=True,
        fold_num: Optional[int] = None,
        verbose=False,
        epsilon=1,
        consistent_bs=True,
        only_continuations=True,
        test_dir="test_outputs",
        score_dir="model_scores",
        gen_dir="model_outputs",
        calc_stats=True,
        noise=0,
        drift=False,
        quiet=True,
    ):
        super().__init__(
            train_cfg,
            net,
            (model_name1, seed1),
            (model_name2, seed2),
            None,
            None,
            1,
            # train_cfg.seed,
        )

        # remove unnecessary attributes
        del self.datagen

        # this is all just for one fold of the distribution data
        self.fold_num = fold_num
        self.metric = metric
        self.only_continuations = only_continuations

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not (self.device == "cuda"):
            logger.warning("CUDA is not available. Using CPU.")
        self.net.to(self.device)

        self.dataset = load_into_scores_ds(
            model_name1,
            seed1,
            model_name2,
            seed2,
            metric,
            fold_num=fold_num,
            only_continuations=only_continuations,
            test_dir=test_dir,
            score_dir=score_dir,
            gen_dir=gen_dir,
            noise=noise,
        )

        # This is the batch size for the network. Should probably ideally be the same as the overall batch size
        self.net_bs = train_cfg.net_batch_size if not consistent_bs else self.bs
        if not (self.net_bs == self.bs):
            logger.warning(
                f"Using different batch size within betting score network (self.net_bs = {self.net_bs}) than for sequences (self.bs = {self.bs}). Might lead to unexpected behavior."
            )

        self.num_batches = (len(self.dataset) + self.bs - 1) // self.bs
        if self.num_batches * self.bs < len(self.dataset):
            logger.warning(
                f"{len(self.dataset) - self.num_batches * self.bs} samples will be discarded as they don't fit into a full batch."
            )

        self.drift = drift
        self.batches, self.batch_indices = self.get_kfold_sequence_batches()
        logger.info(f"Number of sequence batches created: {len(self.batches)}")

        # Epsilon for tolerance test
        self.epsilon = epsilon

        # for logging/tracking
        self.use_wandb = use_wandb
        self.verbose = verbose
        self.current_total_epoch = 0
        self.columns = ["sequence", "samples", "betting_score", "wealth", "test_positive", "epochs", "time"]
        self.data = pd.DataFrame(columns=self.columns)

        # for fast analysis
        self.test_positive = False

        self.calc_stats = calc_stats
        if self.calc_stats:
            self.stat_dict = {
                "mean1": [],
                "mean2": [],
                "std1": [],
                "std2": [],
                "ws": [],
                "ks_p-value": [],
                "fold_number": [],
                "sequence": [],
                "num_samples": [],
                "t_pvalue": [],
                "ad_pvalue": [],
            }
        else:
            self.stat_dict = None

        self.noise = noise

        self.quiet = quiet

    def add_sequence_data(self, sequence, betting_score, wealth, samples, epochs, time_taken):
        """Add a new row of sequence-specific data"""
        new_row = pd.DataFrame(
            [
                {
                    "sequence": sequence,
                    "samples": samples,
                    "betting_score": betting_score,
                    "wealth": wealth,
                    "test_positive": int(self.test_positive),
                    "epochs": epochs,
                    "time": time_taken,
                }
            ]
        )

        self.data = pd.concat([self.data, new_row], ignore_index=True)

    def get_kfold_sequence_batches(self):
        """
        Responsible for dividing the samples in the current fold into sequences
        """
        kf = KFold(n_splits=self.num_batches, shuffle=True, random_state=self.seed)

        valid_size = self.num_batches * self.bs
        logger.info(f"Size of all batches: {valid_size}")
        if valid_size < len(self.dataset):
            rng = np.random.RandomState(self.seed)
            self.dataset = rng.permutation(self.dataset)[:valid_size]
            logger.info(f"Whole dataset has been trimmed to length: {len(self.dataset)}")

        batches = []
        batch_indices_list = []
        for _, batch_indices in kf.split(self.dataset):
            batches.append(Subset(self.dataset, batch_indices))
            batch_indices_list.append(batch_indices)

        if self.drift:
            # add drift to all batches
            num_batches = len(batches)
            drift_per_batch = 0.2 / num_batches
            for i, batch in enumerate(batches):
                batch = list(batch)
                for j in range(len(batch)):
                    batch[j] = (min(1, batch[j][0] + drift_per_batch * i), min(1, batch[j][1] + drift_per_batch * i))
                batches[i] = ScoresDataset(*zip(*batch))

        return batches, batch_indices_list

    def train(self):
        """ """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        betting_scores = []

        # First sequence evaluation
        test_ds = self.batches[0]
        num_samples = len(test_ds)
        test_loader = DataLoader(test_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn)
        _, betting_score = self.train_evaluate_epoch(test_loader, mode="test")
        betting_scores.append(betting_score.item())

        # For first sequence
        self.test_positive = betting_score > (1.0 / self.alpha)
        self.add_sequence_data(
            0,
            betting_score.cpu().item(),
            betting_score.cpu().item(),  # For first sequence, wealth equals betting score
            num_samples,
            0,  # No epochs for first sequence
            0,  # No training time for first sequence
        )

        if self.test_positive:
            logger.info("Reject null at %f", betting_score)
            # Fill remaining sequences with placeholder data
            for remaining_seq in range(1, min(self.seqs, self.num_batches)):
                self.add_sequence_data(
                    remaining_seq,
                    np.nan,  # betting_score
                    np.nan,  # wealth
                    np.nan,  # samples
                    np.nan,  # epochs
                    np.nan,  # time
                )
        else:
            # Split first batch into train/val
            train_ds, val_ds = train_test_split(self.batches[0], test_size=0.2, random_state=self.seed)
            train_loader = DataLoader(train_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn)

            # Iterate over sequences
            for k in tqdm(range(1, min(self.seqs, self.num_batches)), disable=self.quiet):
                sequence_start_time = time.time()

                test_ds = self.batches[k]
                test_loader = DataLoader(test_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn)
                num_samples += len(test_ds)

                for i in range(self.epochs):
                    _, _ = self.train_evaluate_epoch(train_loader)
                    loss_val, _ = self.train_evaluate_epoch(val_loader, mode="val")

                    # Check for early stopping or end of epochs
                    if self.early_stopper.early_stop(loss_val.detach()) or (i + 1) == self.epochs:
                        # Get S_t value on current batch
                        _, betting_score = self.train_evaluate_epoch(test_loader, mode="test")
                        betting_scores.append(betting_score.detach().cpu().item())
                        wealth = np.prod(np.array(betting_scores[self.T :])) if k >= self.T else 1

                        sequence_time = time.time() - sequence_start_time

                        self.test_positive = wealth > (1.0 / self.alpha)
                        self.add_sequence_data(k, betting_score.cpu().item(), wealth, num_samples, i, sequence_time)

                        if self.test_positive:
                            logger.info("Reject null at %f", wealth)
                            # Fill remaining sequences with placeholder data
                            for remaining_seq in range(k + 1, min(self.seqs, self.num_batches)):
                                self.add_sequence_data(
                                    remaining_seq,
                                    np.nan,  # betting_score
                                    np.nan,  # wealth
                                    np.nan,  # samples
                                    np.nan,  # epochs
                                    np.nan,  # time
                                )
                            break

                        # Update datasets for next sequence
                        train_ds = ConcatDataset([train_ds, val_ds])
                        train_loader = DataLoader(train_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn)
                        val_ds = test_ds
                        val_loader = DataLoader(val_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn)
                        break

                # Reset the early stopper for the next sequence
                self.early_stopper.reset()

                if self.test_positive:
                    break

        if not self.test_positive:
            logger.info(f"Null hypothesis not rejected. Final wealth at {wealth}.")

        self.data["fold_number"] = self.fold_num

        if self.calc_stats:
            self.calculate_statistics()
            stat_df = pd.DataFrame(self.stat_dict)
        else:
            stat_df = None

        return self.data, self.test_positive, stat_df

    def train_evaluate_epoch(self, data_loader, mode="train"):
        """ """

        aggregated_loss = 0
        betting_score = 1  # This does not mean we are calculating wealth from scratch, just functions as blank slate for current betting score
        num_samples = len(data_loader.dataset)

        for batch in data_loader:
            tau1, tau2 = torch.split(batch, 1, dim=1)
            tau1 = tau1.to(self.device)
            tau2 = tau2.to(self.device)
            if mode == "train":
                self.net.train()
                # values for tau1 and tau2
                out = self.net(tau1, tau2)
            else:
                self.net.eval()
                out = self.net(tau1, tau2).detach()

            loss = -out.mean() + self.l1_lambda * self.l1_regularization()
            aggregated_loss += -out.sum()  # we can leave epsilon out for optimization

            # need epsilon here for calculating the tolerant betting score
            num_batch_samples = out.shape[0]
            betting_score *= torch.exp(-self.epsilon * num_batch_samples + out.sum())

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return aggregated_loss / num_samples, betting_score

    def calculate_statistics(self):
        """ """
        all_scores1 = [self.dataset.data[i][0] for i in range(len(self.dataset))]
        all_scores2 = [self.dataset.data[i][1] for i in range(len(self.dataset))]

        # Calculate the mean and standard deviation of the scores
        mean1 = np.mean(all_scores1)
        mean2 = np.mean(all_scores2)
        std1 = np.std(all_scores1)
        std2 = np.std(all_scores2)

        # Calculate the Wasserstein distance
        ws = wasserstein_distance(all_scores1, all_scores2)

        scores1 = []
        scores2 = []

        for seq_num in range(0, len(self.batches)):
            # Construct new datasets by concatenating all batches
            batch_indices = self.batch_indices[seq_num]
            scores1 += [self.dataset.data[i][0] for i in batch_indices]
            scores2 += [self.dataset.data[i][1] for i in batch_indices]
            assert len(scores1) == len(scores2), "Length of scores1 and scores2 should be the same"
            num_samples = len(scores1)

            # t-test
            t_pval = ttest_ind(scores1, scores2)

            # KS test
            ks_pval = ks_2samp(scores1, scores2)[1]

            # Anderson-Darling test
            ad_pval = anderson_ksamp([scores1, scores2])[2]

            self.stat_dict["mean1"].append(mean1)
            self.stat_dict["mean2"].append(mean2)
            self.stat_dict["std1"].append(std1)
            self.stat_dict["std2"].append(std2)
            self.stat_dict["ws"].append(ws)
            self.stat_dict["ks_p-value"].append(ks_pval)
            self.stat_dict["t_pvalue"].append(t_pval)
            self.stat_dict["ad_pvalue"].append(ad_pval)
            self.stat_dict["fold_number"].append(self.fold_num)
            self.stat_dict["sequence"].append(seq_num)
            self.stat_dict["num_samples"].append(num_samples)


class OfflineTrainerCombined(OfflineTrainer):
    def __init__(
        self,
        train_cfg,
        net_davt,
        net_c2st,  # Add second network for C2ST
        model_name1,
        seed1,
        model_name2,
        seed2,
        metric="perspective",
        use_wandb=True,
        fold_num: Optional[int] = None,
        verbose=False,
        epsilon=1,
        consistent_bs=True,
        only_continuations=True,
        test_dir="test_outputs",
        score_dir="model_scores",
        gen_dir="model_outputs",
        calc_stats=True,
        noise=0,
        drift=False,
        quiet=True,
    ):
        super().__init__(
            train_cfg,
            net_davt,
            model_name1,
            seed1,
            model_name2,
            seed2,
            metric,
            use_wandb,
            fold_num,
            verbose,
            epsilon,
            consistent_bs,
            only_continuations,
            test_dir,
            score_dir,
            gen_dir,
            calc_stats,
            noise,
            drift,
            quiet=quiet,
        )

        # Add C2ST specific initialization
        self.net_c2st = net_c2st.to(self.device)
        self.loss = torch.nn.CrossEntropyLoss(reduction="sum")
        self.opt_lmbd = 0
        self.run_mean = 0
        self.grad_sq_sum = 1
        self.truncation_level = 0.5

        # Add optimizer for C2ST network
        self.optimizer_c2st = torch.optim.Adam(self.net_c2st.parameters(), lr=train_cfg.lr)

        # Track test positives separately
        self.test_positive_davt = False
        self.test_positive_c2st = False

        # Update columns for the DataFrame to include both DAVT and C2ST metrics
        self.columns = [
            "sequence",
            "samples",
            "betting_score_davt",
            "wealth_davt",
            "betting_score_c2st",
            "wealth_c2st",
            "test_positive_davt",
            "test_positive_c2st",
            "epochs_davt",
            "epochs_c2st",
            "time_davt",
            "time_c2st",
        ]
        self.data = pd.DataFrame(columns=self.columns)

        # Create separate early stoppers for DAVT and C2ST
        self.early_stopper_davt = EarlyStopper(patience=self.patience, min_delta=self.delta)
        self.early_stopper_c2st = EarlyStopper(patience=self.patience, min_delta=self.delta)

        self.stop_davt = False
        self.stop_c2st = False

        self.latest_wealth_davt = -1
        self.latest_wealth_c2st = -1

    def add_sequence_data(
        self,
        sequence,
        betting_score_davt,
        betting_score_c2st,
        wealth_davt,
        wealth_c2st,
        samples,
        epochs_davt,
        epochs_c2st,
        time_davt,
        time_c2st,
    ):
        """Add a new row of sequence-specific data"""
        new_row = pd.DataFrame(
            [
                {
                    "sequence": sequence,
                    "samples": samples,
                    "betting_score_davt": betting_score_davt,
                    "wealth_davt": wealth_davt,
                    "betting_score_c2st": betting_score_c2st,
                    "wealth_c2st": wealth_c2st,
                    "test_positive_davt": int(self.test_positive_davt),
                    "test_positive_c2st": int(self.test_positive_c2st),
                    "epochs_davt": epochs_davt,
                    "epochs_c2st": epochs_c2st,
                    "time_davt": time_davt,
                    "time_c2st": time_c2st,
                }
            ]
        )

        self.data = pd.concat([self.data, new_row], ignore_index=True)

    def train(self):
        """ """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Initialize lists to store betting scores
        betting_scores_davt = []
        betting_scores_c2st = []

        # First sequence evaluation
        test_ds = self.batches[0]
        num_samples = len(test_ds)
        test_loader = DataLoader(test_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn)

        # Evaluate both networks
        _, betting_score_davt = self.train_evaluate_epoch(test_loader, mode="test")
        _, betting_score_c2st = self.train_evaluate_epoch_c2st(test_loader, mode="test")

        betting_scores_davt.append(betting_score_davt.item())
        betting_scores_c2st.append(betting_score_c2st.item())

        # For logging
        self.latest_wealth_davt = betting_score_davt.item()
        self.latest_wealth_c2st = betting_score_c2st.item()

        # Check initial thresholds
        self.test_positive_davt = betting_score_davt > (1.0 / self.alpha)
        self.test_positive_c2st = betting_score_c2st > (1.0 / self.alpha)

        # Add sequence-specific data
        self.add_sequence_data(
            0,
            betting_score_davt.cpu().item(),
            betting_score_c2st.cpu().item(),
            betting_score_davt.cpu().item(),
            betting_score_c2st.cpu().item(),
            num_samples,
            0,
            0,
            0,
            0,
        )

        if not (self.test_positive_davt and self.test_positive_c2st):
            # Split first batch into train/val
            train_ds, val_ds = train_test_split(self.batches[0], test_size=0.2, random_state=self.seed)
            train_loader = DataLoader(train_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn)

            logger.info(f"Starting to iterate through {min(self.seqs, self.num_batches)} sequences.")

            # Loop through sequences
            for k in tqdm(range(1, min(self.seqs, self.num_batches)), disable=self.quiet):
                start_time = time.time()

                self.stop_davt = False
                self.stop_c2st = False

                # Initialize counters and timers for this sequence
                epochs_davt = 0
                epochs_c2st = 0
                time_davt = 0
                time_c2st = 0

                for i in range(self.epochs):
                    test_ds = self.batches[k]
                    test_loader = DataLoader(test_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn)

                    if self.test_positive_davt:
                        betting_score_davt, wealth_davt = np.nan, np.nan
                        self.stop_davt = True
                    elif not self.stop_davt:
                        start_time_davt = time.time()
                        _, _ = self.train_evaluate_epoch(train_loader)
                        loss_val_davt, _ = self.train_evaluate_epoch(val_loader, mode="val")
                        if self.early_stopper_davt.early_stop(loss_val_davt.detach()) or i + 1 == self.epochs:
                            self.stop_davt = True
                            _, betting_score_davt = self.train_evaluate_epoch(test_loader, mode="test")
                            betting_score_davt = betting_score_davt.item()
                            betting_scores_davt.append(betting_score_davt)
                            wealth_davt = np.prod(np.array(betting_scores_davt[self.T :])) if k >= self.T else 1
                            self.latest_wealth_davt = wealth_davt
                            self.test_positive_davt = wealth_davt > (1.0 / self.alpha)
                            epochs_davt = i + 1
                            time_davt = time.time() - start_time_davt
                            if self.test_positive_davt:
                                logger.info(
                                    f"DAVT test positive at sequence {k} with wealth {wealth_davt} and betting score {betting_score_davt}"
                                )

                    if self.test_positive_c2st:
                        betting_score_c2st, wealth_c2st = np.nan, np.nan
                        self.stop_c2st = True
                    elif not self.stop_c2st:
                        start_time_c2st = time.time()
                        _, _ = self.train_evaluate_epoch_c2st(train_loader)
                        loss_val_c2st, _ = self.train_evaluate_epoch_c2st(val_loader, mode="val")
                        if self.early_stopper_c2st.early_stop(loss_val_c2st.detach()) or i + 1 == self.epochs:
                            self.stop_c2st = True
                            _, betting_score_c2st = self.train_evaluate_epoch_c2st(test_loader, mode="test")
                            betting_score_c2st = betting_score_c2st.item()
                            betting_scores_c2st.append(betting_score_c2st)
                            wealth_c2st = np.prod(np.array(betting_scores_c2st[self.T :])) if k >= self.T else 1
                            self.latest_wealth_c2st = wealth_c2st
                            self.test_positive_c2st = wealth_c2st > (1.0 / self.alpha)
                            epochs_c2st = i + 1
                            time_c2st = time.time() - start_time_c2st
                            if self.test_positive_c2st:
                                logger.info(
                                    f"C2ST test positive at sequence {k} with wealth {wealth_c2st} and betting score {betting_score_c2st}"
                                )

                    if self.stop_davt and self.stop_c2st:
                        break
                logger.info(
                    f"Sequence {k} took {round(time.time() - start_time, 3)} seconds. DAVT positive: {self.test_positive_davt} at latest wealth {self.latest_wealth_davt}, C2ST positive: {self.test_positive_c2st} at latest wealth {self.latest_wealth_c2st}."
                )

                num_samples += len(test_ds)

                self.add_sequence_data(
                    k,
                    betting_score_davt,
                    betting_score_c2st,
                    wealth_davt,
                    wealth_c2st,
                    num_samples,
                    epochs_davt,
                    epochs_c2st,
                    time_davt,
                    time_c2st,
                )

                if self.test_positive_davt and self.test_positive_c2st:
                    # Fill remaining sequences with placeholder data
                    for remaining_seq in range(k + 1, min(self.seqs, self.num_batches)):
                        self.add_sequence_data(
                            remaining_seq,
                            np.nan,  # betting_score_davt
                            np.nan,  # betting_score_c2st
                            np.nan,  # wealth_davt
                            np.nan,  # wealth_c2st
                            np.nan,  # num_samples
                            np.nan,  # epochs_davt
                            np.nan,  # epochs_c2st
                            np.nan,  # time_davt
                            np.nan,  # time_c2st
                        )
                    logger.info(f"Both DAVT and C2ST test positive at sequence {k}")
                    break

                # Update datasets for next sequence
                train_ds = ConcatDataset([train_ds, val_ds])
                train_loader = DataLoader(train_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn)
                val_ds = test_ds
                val_loader = DataLoader(val_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn)

                # Reset early stoppers
                self.early_stopper_davt.reset()
                self.early_stopper_c2st.reset()

        if not self.test_positive_davt or not self.test_positive_c2st:
            if not self.test_positive_davt and self.test_positive_c2st:
                logger.info(
                    f"Fold {self.fold_num}: C2ST positive at sequence with final wealth {self.latest_wealth_c2st}, but DAVT test negative with final wealth {self.latest_wealth_davt}."
                )
            if not self.test_positive_c2st and self.test_positive_davt:
                logger.info(
                    f"Fold {self.fold_num}: DAVT positive at sequence with final wealth {self.latest_wealth_davt}, but C2ST test negative with final wealth {self.latest_wealth_c2st}."
                )
            if not self.test_positive_davt and not self.test_positive_c2st:
                logger.info(
                    f"Fold {self.fold_num}: Null hypothesis not rejected by either test. Final wealth: DAVT {self.latest_wealth_davt}, C2ST {self.latest_wealth_c2st}."
                )

        self.data["fold_number"] = self.fold_num

        if self.calc_stats:
            self.calculate_statistics()
            stat_df = pd.DataFrame(self.stat_dict)
        else:
            stat_df = None

        return self.data, (self.test_positive_davt and self.test_positive_c2st), stat_df

    def train_evaluate_epoch_c2st(self, loader, mode="train"):
        """C2ST specific train/evaluate method"""
        aggregated_loss = 0
        e_val, tb_val_ons = 1, 1
        num_samples = len(loader.dataset)

        for batch in loader:
            # Current format: batch is split into tau1 and tau2 along dim=1
            tau1, tau2 = torch.split(batch, 1, dim=1)
            tau1 = tau1.squeeze(1).to(self.device)  # Remove the extra dimension
            tau2 = tau2.squeeze(1).to(self.device)  # Remove the extra dimension

            # Create the two inputs needed for C2ST:
            # z: concatenation of tau1 and tau2
            # tau_z: concatenation of tau2 and tau1 (swapped order)
            z = torch.stack([tau1, tau2], dim=1)
            tau_z = torch.stack([tau2, tau1], dim=1)

            if mode == "train":
                self.net_c2st.train()
                out1 = self.net_c2st(z)
                out2 = self.net_c2st(tau_z)
            else:
                self.net_c2st.eval()
                with torch.no_grad():
                    out1 = self.net_c2st(z)
                    out2 = self.net_c2st(tau_z)

            out = torch.concat((out1, out2))
            # Labels: ones for first half (tau1), zeros for second half (tau2)
            labels = (
                torch.concat((torch.ones((z.shape[0], 1)), torch.zeros((z.shape[0], 1))))
                .squeeze(1)
                .long()
                .to(self.device)
            )
            loss = self.loss(out, labels)
            aggregated_loss += loss

            if mode == "train":
                self.optimizer_c2st.zero_grad()
                loss.backward()
                self.optimizer_c2st.step()

            # Compute C2ST metrics for test mode
            if mode == "test":
                e_val *= self.e_c2st(labels, out.detach())
                p_val, acc = self.s_c2st(labels, out.detach())
                l_val = self.l_c2st(labels, out.detach())
                results_tb = self.testing_by_betting(labels, out.detach())
                tb_val_ons *= results_tb[1]

        return aggregated_loss / num_samples, tb_val_ons

    def testing_by_betting(self, y, logits):
        """Copy implementation from TrainerC2ST"""
        w = 2 * y - 1
        f = torch.nn.Softmax(dim=1)
        ft = 2 * f(logits)[:, 1] - 1
        e_val = torch.exp(torch.sum(torch.log(1 + w * ft)))
        payoffs = w * ft

        grad = self.run_mean / (1 + self.run_mean * self.opt_lmbd)
        self.grad_sq_sum += grad**2
        self.opt_lmbd = max(
            0, min(self.truncation_level, self.opt_lmbd + 2 / (2 - np.log(3)) * grad / self.grad_sq_sum)
        )
        e_val_ons = torch.exp(torch.log(1 + self.opt_lmbd * payoffs.sum()))
        self.run_mean = payoffs.mean()

        return e_val, e_val_ons

    def e_c2st(self, y, logits):
        """Copy implementation from TrainerC2ST"""
        emp_freq_class0 = 1 - (y[y == 1]).sum() / y.shape[0]
        emp_freq_class1 = (y[y == 1]).sum() / y.shape[0]

        f = torch.nn.Softmax(dim=1)
        prob = f(logits)
        pred_prob_class0 = prob[:, 0]
        pred_prob_class1 = prob[:, 1]
        log_eval = torch.sum(
            y * torch.log(pred_prob_class1 / emp_freq_class1) + (1 - y) * torch.log(pred_prob_class0 / emp_freq_class0)
        ).double()
        eval = torch.exp(log_eval)

        return eval

    def s_c2st(self, y, logits):
        """Copy implementation from TrainerC2ST"""
        y_hat = torch.argmax(logits, dim=1)
        n = y.shape[0]
        accuracy = torch.sum(y == y_hat) / n
        stats = np.zeros(500)  # Using default n_per=500
        permutations, n_per = self.first_k_unique_permutations(n, 500)
        for r in range(n_per):
            ind = np.asarray(permutations[r])
            y_perm = y.clone()[ind]
            stats[r] = torch.sum(y_perm == y_hat) / y.shape[0]
        sorted_stats = np.sort(stats)
        p_val = (np.sum(sorted_stats >= accuracy.item()) + 1) / (n_per + 1)

        return p_val, accuracy

    def l_c2st(self, y, logits):
        """Copy implementation from TrainerC2ST"""
        y_hat = torch.argmax(logits, dim=1)
        logit = logits[:, 1] - logits[:, 0]
        n = y.shape[0]
        true_stat = logit[y == 1].mean() - logit[y == 0].mean()
        stats = np.zeros(500)  # Using default n_per=500
        permutations, n_per = self.first_k_unique_permutations(n, 500)
        for r in range(n_per):
            ind = np.asarray(permutations[r])
            logit_perm = logit.clone()[ind]
            stats[r] = logit_perm[y == 1].mean() - logit_perm[y == 0].mean()
        sorted_stats = np.sort(stats)
        p_val = (np.sum(sorted_stats >= true_stat.item()) + 1) / (n_per + 1)

        return p_val

    def first_k_unique_permutations(self, n, k):
        """Copy implementation from TrainerC2ST"""
        if np.log(k) > n * (np.log(n) - 1) + 0.5 * (np.log(2 * np.pi * n)):
            k = n
        unique_perms = set()
        while len(unique_perms) < k:
            unique_perms.add(tuple(np.random.choice(n, n, replace=False)))
        return list(unique_perms), k
