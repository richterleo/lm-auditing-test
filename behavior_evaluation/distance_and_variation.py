import numpy as np

from scipy.stats import kstest
from typing import List


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


if __name__ == "__main__":
    pass
