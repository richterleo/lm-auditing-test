from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Union, Optional

from utils import create_run_string


@dataclass
class EvalArgs:
    metric: Optional[str] = field(default="toxicity")
    dataset_name: Optional[str] = field(default="allenai/real-toxicity-prompts")
    run_name = create_run_string()
    batch_size: Optional[int] = field(
        default=30, metadata={"help": "How many times to go though the whole dataset"}
    )
    model_id: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    device: Optional[str] = field(default="cpu")
    epochs: Optional[int] = field(default=30)
