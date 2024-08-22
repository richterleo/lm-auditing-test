import importlib
import numpy as np
import logging
import random
import torch
import wandb
import sys
import os
import pandas as pd

from collections import defaultdict

from copy import deepcopy
from sklearn.model_selection import train_test_split, KFold
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from transformers import pipeline, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from tqdm import tqdm

# own utilities
from auditing_test.dataloader import ScoresDataset, collate_fn, load_into_scores_ds

# from arguments import Cfg
from utils.generate_and_evaluate import eval_on_metric
from utils.utils import translate_model_kwargs, time_block, NestedKeyDataset, terminator

# deep_anytime_testing = importlib.import_module("deep-anytime-testing")
# train = importlib.import_module("deep-anytime-testing.trainer.trainer")
# Trainer = getattr(train, "Trainer")

orig_models = importlib.import_module(
    "deep-anytime-testing.trainer.trainer", package="deep-anytime-testing"
)
Trainer = getattr(orig_models, "Trainer")

logger = logging.getLogger(__name__)
