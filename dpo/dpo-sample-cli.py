#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

from datasets import load_dataset

from peft import LoraConfig

DEFAULT_SYSTEM_MESSAGE = "You are Pasquale, a helpful AI assistant."

def rec_extract_assistant_messages(messages, index=-1):
  """Recursively extract the last assistant messages from the end of the conversation."""
  res = [messages[index]] if messages[index]["role"] == "assistant" else rec_extract_assistant_messages(messages, index-1)
  return res

def create_triplets(example, tokenizer, default_system_message=DEFAULT_SYSTEM_MESSAGE):
  """Create the triplets (prompt, chosen, rejected)"""
  # Extract the N-1 turns to form the prompt
  # Prepend a system message if the first message is not a system message
  prompt_messages = example["chosen"][:-1]
  if example["chosen"][0]["role"] != "system":
      prompt_messages.insert(0, {"role": "system", "content": default_system_message})
  # Now we extract the final assistant turn to define chosen/rejected responses
  chosen_messages = rec_extract_assistant_messages(example["chosen"])
  rejected_messages = rec_extract_assistant_messages(example["rejected"])
 
  # apply template to the messages and return the triplets
  res = {
     "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False),
     "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False),
     "rejected": tokenizer.apply_chat_template(rejected_messages, tokenize=False)
     }
  
  return res

# Load Tokenizer from the hub
model_id = "cognitivecomputations/dolphin-2.1-mistral-7b" # replace with your model id
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left' # to prevent errors with FA
tokenizer.truncation_side = 'left' # to prevent cutting off last generation

if (not os.path.exists('train.json')) or (not os.path.exists('test.json')):
    # Load dataset from the hub
    ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
    ds = ds.map(create_triplets, remove_columns=ds.features, fn_kwargs={"tokenizer": tokenizer})

    # Keep 6000 instances for testing
    ds = ds.train_test_split(test_size=6000)

    # save datasets to disk
    ds["train"].to_json("train.json", orient="records")
    ds["test"].to_json("test.json", orient="records")

breakpoint()

train_ds = load_dataset("json", data_files="train.json", split="train")
eval_ds = load_dataset("json", data_files="test.json", split="train")

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map="auto",
                                             use_cache=False,
                                             attn_implementation="flash_attention_2",
                                             torch_dtype=torch.bfloat16,
                                             quantization_config=bnb_config)

# Compared to the SFTTrainer the DPOTrainer has two parameter related to dataset sizing with max_prompt_length and max_length. The max_prompt_length is the maximum length of the prompt and the max_length is the maximum length of the prompt + chosen or rejected response. Those are used for tokenization, padding and trunctation. This means if we set those wrongly our data will be potentially cut off, but if we set them too high we will waste memory and time.

max_prompt_length = 1024
max_length = 2048

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(lora_alpha=128,
                         lora_dropout=0.05,
                         r=256,
                         bias="none",
                         target_modules="all-linear",
                         task_type="CAUSAL_LM")

args = TrainingArguments(output_dir="doplhin-dpo",          # directory to save and repository id
                         num_train_epochs=1,                # number of training epochs
                         per_device_train_batch_size=12,    # batch size per device during training
                         per_device_eval_batch_size=4,      # batch size for evaluation
                         gradient_accumulation_steps=1,     # number of steps before performing a backward/update pass
                         gradient_checkpointing=True,       # use gradient checkpointing to save memory
                         optim="adamw_torch_fused",         # use fused adamw optimizer
                         learning_rate=5e-5,                # 10x higher LR than QLoRA paper
                         max_grad_norm=0.3,                 # max gradient norm based on QLoRA paper
                         warmup_ratio=0.1,                  # warmup ratio based on QLoRA paper
                         lr_scheduler_type="cosine",        # use cosine learning rate scheduler
                         logging_steps=25,                  # log every 25 steps
                         save_steps=500,                    # when to save checkpoint
                         save_total_limit=2,                # limit the total amount of checkpoints
                         evaluation_strategy="steps",       # evaluate every 1000 steps
                         eval_steps=700,                    # when to evaluate
                         bf16=True,                         # use bfloat16 precision
                         tf32=True,                         # use tf32 precision
                         push_to_hub=False,                 # push model to hub
                         report_to="tensorboard")           # report metrics to tensorboard

dpo_args = {
   "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
   "loss_type": "sigmoid"                  # The loss type for DPO.
}

trainer = DPOTrainer(model,
                     ref_model=None, # set to none since we use peft
                     peft_config=peft_config,
                     args=args,
                     train_dataset=train_ds,
                     eval_dataset=eval_ds,
                     tokenizer=tokenizer,
                     max_length=max_length,
                     max_prompt_length=max_prompt_length,
                     beta=dpo_args["beta"],
                     loss_type=dpo_args["loss_type"])

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()
 
# save model at the end of training
trainer.save_model()