from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Union, Optional

from utils import create_run_string


@dataclass
class EvalArgs:
    
    metric: Optional[str] = field(default="toxicity")
    dataset_name: Optional[str] = field(default="allenai/real-toxicity-prompts")
    run_name = create_run_string()
    batch_size: Optional[int] = field(default=30, metadata={"help": "How many times to go though the whole dataset"})
    model_name: Optional[str] = field(default="llama")
