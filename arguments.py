from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Union, Optional

from utils import create_run_string


@dataclass
class EvalArgs:
    metric: Optional[str] = field(default="toxicity")
    lower_lim: Optional[float] = field(default=0.0)
    upper_lim: Optional[float] = field(default=1.0)
    dataset_name: Optional[str] = field(default="allenai/real-toxicity-prompts")
    run_name = create_run_string()
    model_id: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    device: Optional[str] = field(default="cuda")
    epochs: Optional[int] = field(
        default=30, metadata={"help": "How many times to go though the whole dataset"}
    )
    num_samples: Optional[int] = field(default=500)
    num_bins: Optional[int] = field(default=10)
    temperature: Optional[float] = field(default=1.0)
    do_sample: Optional[bool] = field(default=True)
    max_length: Optional[int] = field(default=50)


@dataclass
class Cfg:
    @dataclass
    class EarlyStopping:
        patience: Optional[int] = field(default=10)
        delta: Optional[float] = field(default=0.0)

    # Main configuration attributes
    seed: Optional[int] = field(default=42)
    lr: Optional[float] = field(default=0.0005)
    epochs: Optional[int] = field(default=10)
    seqs: Optional[int] = field(default=60)  # number of mini-batches
    alpha: Optional[float] = field(default=0.05)  # significance level
    T: Optional[int] = field(default=0)
    batch_size: Optional[int] = field(default=8)
    save_dir: Optional[str] = field(default="models")
    save: Optional[bool] = field(default=True)
    l1_lambda: Optional[float] = field(default=0.0)
    l2_lambda: Optional[float] = field(default=0.0)

    # Include the early_stopping configuration as a nested attribute
    earlystopping: EarlyStopping = field(default_factory=EarlyStopping)
