import json
import numpy as np
import itertools

from collections import defaultdict
from scipy.stats import kstest



class Variation:

    def __init__(self, data):
        '''
        
        '''
        self.data = data
        self.binned_data = None
        
    def bin(self, num_bins=50, lower_lim=0, upper_lim=1):
        
        count = len(self.data['0'])
        self.binned_data = {key: np.histogram(vals, bins=num_bins, range=(lower_lim, upper_lim))[0]/count for key, vals in self.data.items()}
    
    def calc_all_variation(self, binned_data):
        
        all_var = defaultdict(dict)
        
        for (pkey, qkey) in itertools.combinations(binned_data.keys(), 2):
            
            all_var['tot_var'][(pkey, qkey)] = self._calc_tot_discrete_variation(binned_data[pkey], binned_data[qkey])
            all_var['ks_dist'][(pkey, qkey)] = self._calc_kolmogorov_variation(binned_data[pkey], binned_data[qkey])
            
        return all_var
    
    def calc_tot_variation(self, p, q):
        
        '''
        
        '''
        pass

    def calc_ak_variation(self, p, q):
        '''
        
        '''
        pass
    
    def _calc_kolmogorov_variation(self, p, q):
        '''
        
        '''
        ks_distance = kstest(p, q)
        
        return ks_distance[0]
    
    def _calc_tot_discrete_variation(self, pr, qr):
        
        '''
        '''
        pr = np.array(pr)
        qr = np.array(qr)
        
        tot_diff = np.abs(pr - qr)
        tot_var = tot_diff.sum().item()
        
        return tot_var
        
        
        

if __name__ == "__main__":
    
    file_name = "tox_scores.json"

    with open(file_name) as json_data:
        data = json.load(json_data)
        
    
    variation = Variation(data)
    variation.bin()
        
    tot_var = variation.calc_all_variation(variation.binned_data)
    
    print(tot_var)
