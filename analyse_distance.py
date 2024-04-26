import json
import os


from utils import get_scores_from_wandb
from variation import Variation


if __name__ == "__main__":
    
    
    # file_path = get_scores_from_wandb("nvllytxc")
    
    file_path = "./outputs/nvllytxc/tox_scores.json"
    with open(file_path) as json_data:
        data = json.load(json_data)
    
    variation = Variation(data["tox_scores"])
        
    all_var = variation.calc_all_variation()
    print(all_var)
    
