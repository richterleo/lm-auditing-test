import json
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
import evaluate
import torch 
from tqdm import tqdm
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}.")

reps = 30

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

for rep in tqdm(range(reps)):

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

    tox_scores[rep] = toxicity_ratings['toxicity']
    
print("Finished evaluating.")

file_name = 'tox_scores.json'

with open(file_name, 'w') as file:
    json.dump(tox_scores, file, indent=4)
    
print("Saved scores.")

