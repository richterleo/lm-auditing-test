import numpy as np

from collections import Counter

from variation import Variation, calc_tot_discrete_variation


# def closeness_2samp_unequal_get_m2(m_1, eps=0.1, num_bins=10):
#     """ """
#     assert eps > num_bins ** (
#         -1 / 12
#     ), f"Epsilon must be bigger than n^(-1/12) = {num_bins ** (-1/12):.3g}"
#     assert (
#         m_1 > (num_bins ** (2 / 3)) / (eps ** (4 / 3))
#     ), f"Number of samples from p must be bigger than (n^(2/3))/(epsilon^(4/3)) = {(num_bins ** (2/3)) / (eps ** (4/3))}"


def closeness_testing(
    S1, S2, T1, T2, num_bins=10, epsilon=0.1, epsilon_threshold=1e-7, C0=1
):
    assert (
        epsilon_threshold < epsilon
    ), f"Epsilon cannot be negigibly small (value given = {epsilon}, but lower threshold set to {epsilon_threshold})."
    # TODO: check condition for m_1

    assert (
        S1.shape[0] == S2.shape[0]
    ), f"Samples from p must have the same size, but S1 has {S1.shape[0]} and S2 has {S2.shape[0]} samples."
    assert (
        T1.shape[0] == T2.shape[0]
    ), f"Samples from p must have the same size, but S1 has {S1.shape[0]} and S2 has {S2.shape[0]} samples."

    m1 = S1.shape[0]
    m2 = T1.shape[0]

    b = C0 * np.log(num_bins) / m2

    counts_s1 = np.histogram(S1, bins=num_bins, range=(lower_lim, upper_lim))
    counts_s2 = np.histogram(S2, bins=num_bins, range=(lower_lim, upper_lim))
    counts_t1 = np.histogram(T1, bins=num_bins, range=(lower_lim, upper_lim))
    counts_t2 = np.histogram(T2, bins=num_bins, range=(lower_lim, upper_lim))

    # Determine B set by writing a helper function
    def get_indices(arr, m):
        mask = (arr / m) > b
        indices = np.where(mask)[0]  # Find indices where condition is true

        return indices

    # Find the indices where the condition is True
    B1 = {i for i in range(1, num_bins + 1) if i in get_indices(counts_s1, m1)}
    B2 = {i for i in range(1, num_bins + 1) if i in get_indices(counts_t1, m1)}

    B = B1 | B2

    # Step 1: Check condition (2)
    condition_2 = all(
        abs(counts_S1.get(i, 0) / m1 - counts_S2.get(i, 0) / m2) <= epsilon / 6
        for i in B
    )

    # Step 2: Calculate Z
    Z = sum(
        ((m2 * counts_S1.get(i, 0) - m1 * counts_S2.get(i, 0)) ** 2)
        / (counts_S1.get(i, 0) + counts_S2.get(i, 0))
        for i in range(1, n + 1)
        if i not in B
    )

    C_gamma = 1  # This constant is assumed, needs to be defined based on gamma
    condition_3 = Z <= C_gamma * (m1 ** (-3 / 2)) * (m2**2)

    # Step 3: Check conditions for gamma
    if gamma >= 1 / 9:
        if condition_2 and condition_3:
            return "ACCEPT"
        else:
            return "REJECT"
    else:
        # Step 4: Calculate R and check condition (4)
        C1 = 1  # This constant is assumed
        R = sum(
            1
            for i in range(1, n + 1)
            if counts_S2.get(i, 0) >= 2 and counts_S1.get(i, 0) + 1
        )
        condition_4 = all(
            counts_S2.get(i, 0) < 3
            and counts_S1.get(i, 0) < C2 * (m2**2) / (m1**2) * (n ** (-1 / 3))
            for i in range(1, n + 1)
            if i not in B
        )
        C2 = 1  # This constant is assumed
        if condition_2 and condition_3 and condition_4:
            return "ACCEPT"
        else:
            return "REJECT"


# def closeness_1samp(p, cdf, eps=0.1):
#     """
#     Adapted from scipy, e.g. KstestResult
#     """
#     pass


# def closeness_2samp(p, q, eps=0.1, num_bins=10):
#     """ """
#     # get respective sample sizes
#     n = p.shape[0]
#     m = q.shape[0]

#     #


# def closeness_2samp_unequal(p, q, eps=0.1, num_bins=10):
#     """ """
#     pass
