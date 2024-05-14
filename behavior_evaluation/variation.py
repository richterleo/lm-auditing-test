import json
import numpy as np
import itertools

from collections import defaultdict
from scipy.stats import kstest
from typing import List


class VariationSampler:
    def __init__(self, data):
        """ """
        self.data = data

    def bin_data(self, num_bins=9, lower_lim=0, upper_lim=1):
        count = len(self.data["0"])

        hist_data = {
            key: np.histogram(vals, bins=num_bins, range=(lower_lim, upper_lim))[0]
            / count
            for key, vals in self.data.items()
        }

        return hist_data

    def calc_all_variation(self, var_style="tot_var"):
        """ """

        all_var = defaultdict(dict)

        hist_data = self.bin_data()

        for pkey, qkey in itertools.combinations(hist_data.keys(), 2):
            if var_style == "tot_var":
                all_var["tot_var"][(pkey, qkey)] = calc_tot_discrete_variation(
                    hist_data[pkey], hist_data[qkey]
                )

            elif var_style == "ks":
                all_var["ks_dist"][(pkey, qkey)] = calc_kolmogorov_variation(
                    hist_data[pkey], hist_data[qkey]
                )

        return all_var


def calc_ak_variation(self, p, q):
    """ """
    pass


def calc_kolmogorov_variation(self, p, q):
    """ """
    ks_distance = kstest(p, q)

    return ks_distance[0]


def calc_tot_discrete_variation(pr, qr):
    """ """
    pr = np.array(pr)
    qr = np.array(qr)

    tot_diff = np.abs(pr - qr)
    tot_var = tot_diff.sum().item()

    return tot_var


def empirical_wasserstein(psamp: List, qsamp: List, p: float = 1) -> float:
    """
    Computes the empirical Wasserstein distance between two samples of probability distributions.

    Args:
        psamp (List): Sample of probability distribution 1 of size n.
        qsamp (List): Sample of probability distribution 2 of size m.
        p (float): Order of the Wasserstein distance. Must be greater than or equal to 1.

    Returns:
        wasserstein_dist (float): Empirical Wasserstein distance between the two samples.

    """
    assert (
        p >= 1
    ), "p must be greater than or equal to 1"  # TODO: implement other p-values

    # Sort the samples
    sorted_samples1 = np.sort(psamp)
    sorted_samples2 = np.sort(qsamp)

    # Combine and sort all samples
    combined_samples = np.sort(np.concatenate([sorted_samples1, sorted_samples2]))

    # Compute empirical CDFs at each point in the combined sample list
    cdf1 = np.searchsorted(sorted_samples1, combined_samples, side="right") / len(psamp)
    cdf2 = np.searchsorted(sorted_samples2, combined_samples, side="right") / len(qsamp)

    # Calculate the Wasserstein distance
    wasserstein_dist = np.sum(
        np.abs(cdf1 - cdf2) * np.diff(np.concatenate(([0], combined_samples)))
    )

    return wasserstein_dist
