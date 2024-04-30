import json
import numpy as np
import itertools

from collections import defaultdict
from scipy.stats import kstest



class VariationSampler:

    def __init__(self, data):
        '''
        
        '''
        self.data = data
        
    def bin_data(self, num_bins=9, lower_lim=0, upper_lim=1):
        
        count = len(self.data['0'])
        
        hist_data = {key: np.histogram(vals, bins=num_bins, range=(lower_lim, upper_lim))[0]/count for key, vals in self.data.items()}
        
        return hist_data
        
        
    
    def calc_all_variation(self, var_style="tot_var"):
        '''
        
        '''
        
        all_var = defaultdict(dict)
        
        hist_data = self.bin_data()
        
        for (pkey, qkey) in itertools.combinations(hist_data.keys(), 2):
            
            if var_style == "tot_var":
                all_var['tot_var'][(pkey, qkey)] = calc_tot_discrete_variation(hist_data[pkey], hist_data[qkey])
                
            elif var_style == "ks":
                all_var['ks_dist'][(pkey, qkey)] = calc_kolmogorov_variation(hist_data[pkey], hist_data[qkey])
            
        return all_var


def calc_ak_variation(self, p, q):
    '''
    
    '''
    pass
    


def calc_kolmogorov_variation(self, p, q):
    '''
    
    '''
    ks_distance = kstest(p, q)
    
    return ks_distance[0]
        
    
def calc_tot_discrete_variation(pr, qr):
    
    '''
    '''
    pr = np.array(pr)
    qr = np.array(qr)
    
    tot_diff = np.abs(pr - qr)
    tot_var = tot_diff.sum().item()
    
    return tot_var
        
        
        

# if __name__ == "__main__":
    
#     file_name = "tox_scores.json"

#     with open(file_name) as json_data:
#         data = json.load(json_data)
        
    
#     variation = Variation(data)
#     variation.bin()
        
#     tot_var = variation.calc_all_variation(variation.binned_data)
    
#     print(tot_var)
