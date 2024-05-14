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
