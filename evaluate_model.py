import json
import os
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
import evaluate
import torch 
from tqdm import tqdm
import random
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}.")

os.environ["WANDB_API_KEY"] = "1c84a4abed1d390fbe37478c7cb82a84e4650881"
os.environ["WANDB_LOG_LEVEL"] = "debug"
wandb.init(project="toxicity_evaluation", entity="richter-leo94")

epochs = 30

toxicity_prompts = load_dataset("allenai/real-toxicity-prompts", split="train")

text_generation = pipeline("text-generation", model="gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("Start evaluating.")

def get_random_prompts(dataset, num_examples=500):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    return(dataset[picks])

toxic_sample= get_random_prompts(toxicity_prompts)
toxic_prompts = [p['text'] for p in toxic_sample['prompt']]

print(f"Number of prompts: {len(toxic_prompts)}.")
tox_scores = {}

for epoch in tqdm(range(epochs)):

    model_continuations=[]
    for prompt in tqdm(toxic_prompts):
        generation = text_generation(prompt, 
                                     max_length=50, 
                                     do_sample=True,
                                     temperature=2.0, 
                                     pad_token_id=50256)
        
        continuation = generation[0]['generated_text'].replace(prompt,'')
        model_continuations.append(continuation)

    toxicity = evaluate.load("toxicity")

    toxicity_ratings = toxicity.compute(predictions=model_continuations)

    wandb.log({"Rep": epoch, "Toxicity scores": toxicity_ratings['toxicity']})
    
    tox_scores[epochs] = toxicity_ratings['toxicity']
    
print("Finished evaluating.")

file_name = 'tox_scores.json'

with open(file_name, 'w') as file:
    json.dump(tox_scores, file, indent=4)
    
print("Saved scores.")

