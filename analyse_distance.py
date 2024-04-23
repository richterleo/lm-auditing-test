import json
import os

from utils import get_scores_from_wandb
from variation import Variation


if __name__ == "__main__":
    
    
    file_path = get_scores_from_wandb("nvllytxc")
    
    
    with open(file_path) as json_data:
        data = json.load(json_data)
        
    
    variation = Variation(data["tox_scores"])
    variation.bin()
        
    tot_var = variation.calc_all_variation(variation.binned_data)
    
    print(tot_var)
    
