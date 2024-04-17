import json
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
import evaluate
import torch 
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

reps = 100

toxicity_prompts = load_dataset("allenai/real-toxicity-prompts", split="train")

text_generation = pipeline("text-generation", model="gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

toxic_prompts = [p['text'] for p in toxicity_prompts['prompt']]
tox_scores = {}

for rep in tqdm(range(reps)):

    model_continuations=[]
    for prompt in toxic_prompts:
        generation = text_generation(prompt, 
                                     max_length=50, 
                                     do_sample=True,
                                     temperature=2, 
                                     pad_token_id=50256)
        
        continuation = generation[0]['generated_text'].replace(prompt,'')
        model_continuations.append(continuation)

    toxicity = evaluate.load("toxicity")

    toxicity_ratings = toxicity.compute(predictions=model_continuations)

    tox_scores[rep] = toxicity_ratings['toxicity']

