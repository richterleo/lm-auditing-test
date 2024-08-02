import importlib
import numpy as np
import os
import torch
import sys

from scipy.stats import kstest, wasserstein_distance
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from arguments import TrainCfg
from utils.utils import initialize_from_config, time_block, load_config
from dah_testing.dataloader import ScoresDataset, collate_fn
from behavior_evaluation.neural_net_distance import CMLP


# Add the submodule and models to the path for eval_trainer
submodule_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "deep-anytime-testing"))
models_path = os.path.join(submodule_path, "models")

for path in [submodule_path, models_path]:
    if path not in sys.path:
        sys.path.append(path)
        


def get_hist_distribution(data, num_bins=9, lower_lim=0, upper_lim=1):
    """ """
    count = len(data)
    hist_data = (
        np.histogram(data, bins=num_bins, range=(lower_lim, upper_lim))[0] / count
    )

    return hist_data


def calc_ak_variation(self, p, q):
    """ """
    pass


def kolmogorov_variation(samples1, samples2):
    """ """
    ks_distance = kstest(samples1, samples2)

    return ks_distance[0]


def calc_tot_discrete_variation(pr, qr):
    """ """
    pr = np.array(pr)
    qr = np.array(qr)

    tot_diff = np.abs(pr - qr)
    tot_var = tot_diff.sum().item()

    return tot_var


def empirical_quantile_function(samples, q):
    """
    Calculate the q-th quantile (inverse CDF) of the given samples.

    Parameters:
    - samples: An array of samples from the distribution.
    - q: A quantile value between 0 and 1.

    Returns:
    - The q-th quantile value.
    """
    sorted_samples = np.sort(samples)
    n = len(samples)

    # Calculate the rank of the quantile
    rank = q * (n - 1)

    # If rank is an integer, return the value at that rank
    if rank.is_integer():
        return sorted_samples[int(rank)]
    else:
        # Interpolation between two surrounding values
        lower_index = int(np.floor(rank))
        upper_index = int(np.ceil(rank))
        lower_value = sorted_samples[lower_index]
        upper_value = sorted_samples[upper_index]
        return lower_value + (upper_value - lower_value) * (rank - lower_index)


def empirical_wasserstein_distance_p1(samples1: List, samples2: List) -> float:
    """
    Computes the empirical Wasserstein distance between two samples of probability distributions.

    Args:
        samples1 (List): Sample of probability distribution 1 of size n.
        samples2 (List): Sample of probability distribution 2 of size m.

    Returns:
        wasserstein_dist (float): Empirical Wasserstein distance between the two samples.

    """

    # Sort the samples
    sorted_samples1 = np.sort(samples1)
    sorted_samples2 = np.sort(samples2)

    # Combine and sort all samples
    combined_samples = np.sort(np.concatenate([sorted_samples1, sorted_samples2]))

    # Compute empirical CDFs at each point in the combined sample list
    cdf1 = np.searchsorted(sorted_samples1, combined_samples, side="right") / len(
        samples1
    )
    cdf2 = np.searchsorted(sorted_samples2, combined_samples, side="right") / len(
        samples2
    )

    # Calculate the Wasserstein distance
    wasserstein_dist = np.sum(
        np.abs(cdf1 - cdf2) * np.diff(np.concatenate(([0], combined_samples)))
    )

    return wasserstein_dist


def empirical_wasserstein_distance(samples1, samples2, p=1):
    """
    Calculate the p-Wasserstein distance between two sets of samples.

    Parameters:
    - samples1: An array of samples from the first distribution.
    - samples2: An array of samples from the second distribution.
    - p: The order of the Wasserstein distance (default is 1).

    Returns:
    - The p-Wasserstein distance between the two distributions.
    """

    assert (
        p >= 1
    ), "The order of the Wasserstein distance must be greater than or equal to 1."

    if p == 1:
        wasserstein_dist = empirical_wasserstein_distance_p1(samples1, samples2)
    else:
        n = len(samples1)
        m = len(samples2)

        quantiles = np.linspace(0, 1, max(n, m), endpoint=True)

        q1 = np.array([empirical_quantile_function(samples1, q) for q in quantiles])
        q2 = np.array([empirical_quantile_function(samples2, q) for q in quantiles])
        wasserstein_dist = (np.sum(np.abs(q1 - q2) ** p * (1 / len(quantiles)))) ** (
            1 / p
        )

    return wasserstein_dist


class NeuralNetDistance:
    def __init__(self, net_cfg, samples1, samples2, train_cfg, random_seed=0, epochs=100, net_type="MMDEMLP"):
        """
        Initializes the NeuralNetDistance object.

        Args:
        - net_cfg (dict): Configuration dictionary for the neural network.
        - samples1 (List): List of samples from distribution 1.
        - samples2 (List): List of samples from distribution 2.
        - train_cfg (dict): Configuration dictionary for training the neural network.
        - random_seed (int): Random seed for reproducibility.
        """

        # train params
        self.random_seed = random_seed
        self.epochs = epochs
        self.lr = train_cfg.lr
        self.patience = train_cfg.earlystopping.patience
        self.delta = train_cfg.earlystopping.delta
        self.alpha = train_cfg.alpha
        self.net_bs = train_cfg.net_batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # L1 and L2 regularization parameters
        self.weight_decay = train_cfg.l2_lambda
        self.l1_lambda = train_cfg.l1_lambda
        
        # initialize network
        # self.net = initialize_from_config(net_config["net"], net_type=net_type)
        # TODO: put this somewhere else
        self.net = CMLP(
            net_cfg["net"]["input_size"],
            net_cfg["net"]["hidden_layer_size"],
            1,
            net_cfg["net"]["layer_norm"],
            False,
            0.4,
            net_cfg["net"]["bias"],
        )
        self.net.to(self.device)

        # Initialize optimizer and early stopper
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        models = importlib.import_module(
        "deep-anytime-testing.models.earlystopping", package="deep-anytime-testing"
            )
        EarlyStopper = getattr(models, "EarlyStopper")
        self.early_stopper = EarlyStopper(patience=self.patience, min_delta=self.delta)
        

        # samples from distribution 1 and 2
        self.samples1 = samples1
        self.samples2 = samples2
        
    def l1_regularization(self):
        l1_regularization = torch.tensor(0., requires_grad=True)
        for name, param in self.net.named_parameters():
            if 'bias' not in name:
                l1_regularization = l1_regularization + torch.norm(param, p=1)
        return l1_regularization
        
    def train(self):
        
        train_val_ds, test_ds = train_test_split(
            ScoresDataset(self.samples1, self.samples2), test_size=0.2, random_state=self.random_seed
        )

        train_ds, val_ds = train_test_split(
            train_val_ds, test_size=0.2, random_state=self.random_seed
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.net_bs, shuffle=True, collate_fn=collate_fn
        )
        
        for epoch in tqdm(range(self.epochs)):
            self.train_evaluate_epoch(train_loader)
            val_loss = self.train_evaluate_epoch(val_loader, mode="val")
            

            # Check for early stopping or end of epochs
            if (
                self.early_stopper.early_stop(val_loss.detach())
                or (epoch + 1) == self.epochs
            ):
                neural_net_distance = self.train_evaluate_epoch(test_loader, mode="test")

                return neural_net_distance
            
    def train_evaluate_epoch(self, data_loader, mode="train"):
        """
        Train or evaluate the neural network for one epoch.

        Args:
        - data_loader (DataLoader): DataLoader object for the dataset.
        - mode (str): Indicates if the network is in training or evaluation mode.

        Returns:
        - aggregated_loss (float): The aggregated loss over the epoch.
        """

        aggregated_loss = 0
        aggregated_diff = 0
        num_samples = len(data_loader.dataset)
        num_samples_control = 0
        

        for i, batch in enumerate(data_loader):
            if mode == "test":
                print(f"batch number: {i}")
                print(f"This batch has length {len(batch)}")
            tau1, tau2 = torch.split(batch, 1, dim=1)
            tau1 = tau1.to(self.device)
            tau2 = tau2.to(self.device)
            
            if mode == "train":
                self.net.train()
                out = self.net(tau1, tau2)
            else:
                self.net.eval()
                out = self.net(tau1, tau2).detach()

            loss = -out.mean() + self.l1_lambda * self.l1_regularization()
            aggregated_loss += -out.sum()    

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            elif mode == "test":
                num_samples_control += out.shape[0]
                aggregated_diff += torch.exp(out).sum()
                

        if mode == "test":
            # 1 + g(X)- g(Y) -1 = g(X) - g(Y)
            print(f"num_samples control: {num_samples_control}, num_samples: {num_samples}")
            distance = (aggregated_diff / num_samples) - 1
            return distance
        
        else:
            return aggregated_loss / num_samples

    

    
    


if __name__ == "__main__":
    samples_N01 = np.random.normal(0, 1, 100000)

    # Generate 100,000 samples from N(0.5, 1)
    samples_N05_1 = np.random.normal(0.01, 1, 100000)

    net_cfg = load_config("config.yml")
    train_cfg = TrainCfg()
    
    
    w_dist = wasserstein_distance(samples_N01, samples_N05_1)
    nn_dist_class = NeuralNetDistance(net_cfg, samples_N01, samples_N05_1, train_cfg)
    nn_dist = nn_dist_class.train()
    
    print(f"This is the empirical wasserstein distance: {w_dist}")
    print(f"This is the neural net distance {nn_dist}")
    
    
    
