import numpy as np

from collections import Counter

from behavior_evaluation.variation import Variation, calc_tot_discrete_variation


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
    bin_first=True,  # TODO: implement alternative where data comes binned already
):
    """
    Implementation of closeness test for two samples with unequal sample sizes.
    Algorithm adapted from https://proceedings.neurips.cc/paper_files/paper/2015/hash/5cce8dede893813f879b873962fb669f-Abstract.html

    """

    assert (
        epsilon_threshold < epsilon
    ), f"Epsilon cannot be negigibly small (value given = {epsilon}, but lower threshold set to {epsilon_threshold})."
    # TODO: check condition for m_1

    # Assume that samples just come as arrays of values
    if bin_first:
        S1 = np.histogram(
            S1, bins=num_bins, range=(lower_lim, upper_lim)
        )[
            0
        ]  # TODO: check what the output of the histogram is again. I think it's a tuple and we only need one part
        S2 = np.histogram(S2, bins=num_bins, range=(lower_lim, upper_lim))[0]
        T1 = np.histogram(T1, bins=num_bins, range=(lower_lim, upper_lim))[0]
        T2 = np.histogram(T2, bins=num_bins, range=(lower_lim, upper_lim))[0]

    m1 = S1.sum().item()
    m2 = T1.sum().item()

    assert (
        m1 == S2.sum().item()
    ), f"Samples from p must be the same size, but S1 has {m1} and S2 has {S2.sum().item()} samples."
    assert (
        m2 == T2.sum().item()
    ), f"Samples from p must be the same size, but T1 has {m2} and T2 has {T2.sum().item()} samples."

    num_bins = S1.shape[0]

    b = C0 * np.log(num_bins) / m2

    # Determine B set by writing a helper function
    def get_indices(arr, m):
        mask = (arr / m) > b
        indices = np.where(mask)[0]  # Find indices where condition is true

        return indices

    # Find the indices where the condition is True
    B1 = {i for i in range(1, num_bins + 1) if i in get_indices(S1, m1)}
    B2 = {i for i in range(1, num_bins + 1) if i in get_indices(T1, m1)}

    B = B1 | B2
    mask = np.ones(num_bins)

    # create mask for checking conditions
    mask[list(B)] = 0

    # Check condition (2)
    cond1_value = np.sum(np.abs(S2 / m1 - T2 / m2))
    cond1_met = cond1_value <= epsilon / 6

    # Check condition (3)
    Z_vec = ((m2 * S2 - m1 * T2) ** 2 - (m2**2 * S2 + m1**2 * T2)) / (S2 + T2)
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
        R_vec = (T2 == 2) / (S2 + 1)
        R_vec[mask] = 0
        R = np.sum(R_vec).item()

        cond3_met = R <= C1 * m2**2 / m1

        if cond1_met and cond2_met and cond3_met:
            maskY = T2 >= 3
            maskX = S2 <= C2 * m1 / (m2 * num_bins ** (1 / 3))

            mask = maskY & maskX

            cond4_met = np.sum(mask) == 0

            if cond4_met:
                return "ACCEPT"

        else:
            return "REJECT"


def ctest_2samp_unequal_nonextreme(
    S1,
    S2,
    T1,
    T2,
    gamma=0.1,  # TODO define gamma
    num_bins=10,
    epsilon=0.1,
    epsilon_threshold=1e-7,
    C_gamma=1,  # TODO: define C_gamma
    lower_lim=0,
    upper_lim=1,
    bin_first=True,  # TODO: implement alternative where data comes binned already
):
    """
    Implementation of closeness test for two samples with unequal sample sizes.
    Algorithm adapted from https://proceedings.neurips.cc/paper_files/paper/2015/hash/5cce8dede893813f879b873962fb669f-Abstract.html

    """

    assert (
        epsilon_threshold < epsilon
    ), f"Epsilon cannot be negigibly small (value given = {epsilon}, but lower threshold set to {epsilon_threshold})."
    # TODO: check condition for m_1

    # Assume that samples just come as arrays of values
    if bin_first:
        S1 = np.histogram(
            S1, bins=num_bins, range=(lower_lim, upper_lim)
        )[
            0
        ]  # TODO: check what the output of the histogram is again. I think it's a tuple and we only need one part
        S2 = np.histogram(S2, bins=num_bins, range=(lower_lim, upper_lim))[0]
        T1 = np.histogram(T1, bins=num_bins, range=(lower_lim, upper_lim))[0]
        T2 = np.histogram(T2, bins=num_bins, range=(lower_lim, upper_lim))[0]

    m1 = S1.sum().item()
    m2 = T1.sum().item()

    assert (
        m1 == S2.sum().item()
    ), f"Samples from p must be the same size, but S1 has {m1} and S2 has {S2.sum().item()} samples."
    assert (
        m2 == T2.sum().item()
    ), f"Samples from p must be the same size, but T1 has {m2} and T2 has {T2.sum().item()} samples."

    num_bins = S1.shape[0]

    b2 = (256 * np.log(num_bins)) / (m1)
    b1 = b2 / epsilon**2

    # Determine heavy set by writing a helper function
    def get_indices_heavy_set(arr, m):
        mask = (arr / m) > b1
        indices = np.where(mask)[0]  # Find indices where condition is true

        return indices

    # Define the heavy set and create mask
    B1 = {i for i in range(1, num_bins + 1) if i in get_indices_heavy_set(S1, m1)}
    B2 = {i for i in range(1, num_bins + 1) if i in get_indices_heavy_set(T1, m1)}

    B = B1 | B2
    Bmask = np.ones(num_bins)
    Bmask[~list(B)] = 0

    # Define the medium set and create mask
    Mmask = np.ones(num_bins)
    upper_cond = np.max(S1 / m1, T1 / m2) > b1
    lower_cond = np.max(S1 / m1, T1 / m2) < b2
    combined_cond = upper_cond | lower_cond
    Mmask[combined_cond] = 0

    # Define the light set and create mask
    Hmask = np.ones(num_bins)
    Hmask[Bmask == 0 or Mmask == 0] = 0

    # Check condition 1 ((6) in paper)
    VB_vec = np.abs(S2 / m1 - T2 / m2)
    VB = VB_vec[Bmask].sum().item()
    cond1_met = VB <= epsilon / 6

    # Check condition 2 ((7) in paper)
    WM_vec = (m2 * S2 - m1 * T2) ** 2 - (m2**2 * S2 + m1**2 * T2)
    WM = WM_vec[Mmask].sum().item()
    cond2_met = WM <= (epsilon**2 * m1**2 * m2 * np.log(num_bins)) / 2

    # Check condition 3 ((8) in paper
    ZH_vec = ((m2 * S2 - m1 * T2) ** 2 - (m2**2 * S2 + m1**2 * T2)) / (S2 + T2)
    ZH = ZH_vec[Hmask].sum().item()
    cond3_met = ZH <= C_gamma * m1 ** (3 / 2) * m2

    if cond1_met and cond2_met and cond3_met:
        return "ACCEPT"
    else:
        return "REJECT"


def ctest_2samp_unequal_estimate_m2(
    m1,
    epsilon=0.1,
    num_bins=10,
):
    """ """
    assert (
        m1 >= (num_bins ** (2 / 3)) / (epsilon ** (4 / 3))
    ), f"Number of samples from p must be bigger than (n^(2/3))/(epsilon^(4/3)) = {(num_bins**(2/3))/(epsilon**(4/3))}"
    assert epsilon > num_bins ** (
        -1 / 12
    ), f"Epsilon must be bigger than n^(-1/12) = {num_bins**(-1/12):.3g}"

    ref = np.max(
        num_bins / (m1 ** (1 / 2) * epsilon**2), num_bins ** (1 / 2) / epsilon**2
    )

    return ref


def ctest_1samp(
    X, P, epsilon=0.1, num_bins=10, lower_lim=0, upper_lim=1, bin_first=True
):
    """ """

    if bin_first:
        X = np.histogram(X, bins=num_bins, range=(lower_lim, upper_lim))[0]

    k = X.sum()  # Total number of samples
    num_bins = X.shape[0]

    s = np.min(np.where(np.cumsum(np.sort(P))[::-1] <= epsilon / 8))
    M = range(2, s + 1)
    S = range(s + 1, num_bins + 1)

    # Condition 1
    pm = P[
        2 : s + 1
    ]  # remove item with largest probability and the items with the cumulatively smallest probability (below epsilon/8)

    cond1_vec = ((X - k * P) ** 2 - X) / (P ** (2 / 3))
    cond1_value = cond1_vec[M].sum().item()

    cond1_met = cond1_value > 4 * k * np.sqrt(pm ** (2 / 3))

    if cond1_met:
        return "REJECT"

    else:
        # Condition 2
        cond2_value = X[S].sum().item()
        cond2_met = cond2_value > (3 / 16) * k * epsilon

        if cond2_met:
            return "REJECT"

        else:
            return "ACCEPT"


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
