import numpy as np

from collections import Counter

from variation import Variation, calc_tot_discrete_variation


def ctest_2samp_unequal(
    S1,
    S2,
    T1,
    T2,
    gamma=0.1,  # TODO define gamme
    num_bins=10,
    epsilon=0.1,
    epsilon_threshold=1e-7,
    C0=1,  # TODO: define C0
    C1=1,  # TODO: define C1
    C2=1,  # TODO: define C2
    C_gamma=1,  # TODO: define C_gamma
    lower_lim=0,
    upper_lim=1,
):
    """
    Implementation of closeness test for two samples with unequal sample sizes.
    Algorithm adapted from https://proceedings.neurips.cc/paper_files/paper/2015/hash/5cce8dede893813f879b873962fb669f-Abstract.html

    """

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
    mask = np.ones(num_bins)

    # create mask for checking conditions
    mask[list(B)] = 0

    # Check condition (2)
    cond1_value = np.sum(np.abs(counts_s2 / m1 - counts_t2 / m2))
    cond1_met = cond1_value <= epsilon / 6

    # Check condition (3)
    Z_vec = (
        (m2 * counts_s2 - m1 * counts_t2) ** 2 - (m2**2 * counts_s2 + m1**2 * counts_t2)
    ) / (counts_s2 + counts_t2)
    Z_vec[mask] = 0
    Z = np.sum(Z_vec).item()

    C_gamma = 1  # This constant is assumed, needs to be defined based on gamma
    cond2_met = Z <= C_gamma * (m1 ** (3 / 2)) * m2

    # Step 3: Check conditions for gamma
    if gamma >= 1 / 9:
        if cond1_met and cond2_met:
            return "ACCEPT"
        else:
            return "REJECT"
    else:
        R_vec = (counts_t2 == 2) / (counts_s2 + 1)
        R_vec[mask] = 0
        R = np.sum(R_vec).item()

        cond3_met = R <= C1 * m2**2 / m1

        if cond1_met and cond2_met and cond3_met:
            maskY = counts_t2 >= 3
            maskX = counts_s2 <= C2 * m1 / (m2 * num_bins ** (1 / 3))

            mask = maskY & maskX

            cond4_met = np.sum(mask) == 0

            if cond4_met:
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


# def closeness_2samp_unequal_get_m2(m_1, eps=0.1, num_bins=10):
#     """ """
#     assert eps > num_bins ** (
#         -1 / 12
#     ), f"Epsilon must be bigger than n^(-1/12) = {num_bins ** (-1/12):.3g}"
#     assert (
#         m_1 > (num_bins ** (2 / 3)) / (eps ** (4 / 3))
#     ), f"Number of samples from p must be bigger than (n^(2/3))/(epsilon^(4/3)) = {(num_bins ** (2/3)) / (eps ** (4/3))}"
