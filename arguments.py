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
