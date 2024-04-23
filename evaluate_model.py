import json
import os
import numpy as np

from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
import evaluate
import torch 
from tqdm import tqdm
import random
import wandb

from utils import create_run_string, get_random_prompts, log_scores

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}.")

run_name = create_run_string()
print(f"Run name: {run_name}")

os.environ["WANDB_API_KEY"] = "1c84a4abed1d390fbe37478c7cb82a84e4650881"
os.environ["WANDB_LOG_LEVEL"] = "debug"
wandb.init(project="toxicity_evaluation", entity="richter-leo94", name=run_name)

epochs = 30

toxicity_prompts = load_dataset("allenai/real-toxicity-prompts", split="train")

text_generation = pipeline("text-generation", model="gpt2") # Make this faster using flash attention
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("Start evaluating.")

toxic_sample= get_random_prompts(toxicity_prompts)
toxic_prompts = [p['text'] for p in toxic_sample['prompt']]
print(f"Number of prompts: {len(toxic_prompts)}.")

tox_scores = {}

toxicity = evaluate.load("toxicity")

for epoch in tqdm(range(epochs)):

    model_continuations=[]
    for prompt in tqdm(toxic_prompts):
        generation = text_generation(prompt, 
                                     max_length=100, 
                                     do_sample=True,
                                     temperature=2.0, 
                                     pad_token_id=50256)
        
        continuation = generation[0]['generated_text'].replace(prompt,'')
        model_continuations.append(continuation)


    toxicity_ratings = toxicity.compute(predictions=model_continuations)
    
    #hist = np.histogram(toxicity_ratings['toxicity'])
    
    # table = wandb.Table(data=toxicity_ratings['toxicity'], columns=["scores"])
    # wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title="Histogram")})

    
    tox_scores[epoch] = toxicity_ratings['toxicity']
    print(tox_scores[epoch])
    
    hist = np.histogram(tox_scores[epoch], bins='auto')
    wandb.log({f"Epoch {epoch+1} Toxicity Scores": wandb.Histogram(np_histogram=hist)})
    
    log_scores(tox_scores)
    
    file_name = 'tox_scores.json'

    with open(file_name, 'w') as file:
        json.dump(tox_scores, file, indent=4)
    
    print(f"Saved scores epoch {epoch} out of {epochs}.")
    
print("Finished evaluating.")

wandb.finish()



