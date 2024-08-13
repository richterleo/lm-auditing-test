from dataclasses import dataclass, field
from typing import List, Dict, Optional

from utils.utils import create_run_string


@dataclass
class MetricCfg:
    behavior: str = field(
        default="toxicity", metadata={"help": "Which behaviour to evaluate"}
    )
    metric: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which metric to use. If none is given, the metric is loaded using the behavior string."
        },
    )
    metric_lower_lim: float = field(
        default=0.0,
        metadata={"help": "Lower limit of metric to use to evaluate behavior."},
    )
    metric_upper_lim: float = field(
        default=1.0,
        metadata={"help": "Upper limit of metric to use to evaluate behavior."},
    )
    dataset_name: str = field(
        default="allenai/real-toxicity-prompts",
        metadata={"help": "Name of dataset to use for evaluation."},
    )


@dataclass
class EvalArgs:
    epochs: int = field(
        default=30, metadata={"help": "How many times to go though the whole dataset"}
    )
    num_samples: int = field(
        default=500, metadata={"help": "How many samples to select from dataset"}
    )
    num_bins: int = field(
        default=10, metadata={"help": "How many bins to divide the metric into"}
    )


@dataclass
class LoggingCfg:
    use_wandb: bool = field(
        default=True, metadata={"help": "Whether to use wandb for logging."}
    )
    entity: Optional[str] = field(default="richter-leo94")
    run_name: str = field(
        default=create_run_string(),
        metadata={"help": "Name of run for logging on wandb."},
    )


@dataclass
class ModelCfg:
    model_id: str = field(default="meta-llama/Meta-Llama-3-8B")
    gen_batch_size: int = field(
        default=8, metadata={"help": "Batch size for generation."}
    )

    @dataclass
    class model_kwargs:
        torch_dtype: str = field(
            default="torch.bfloat16",
            metadata={"help": "Patience in training algorithm."},
        )
        load_in_4bit: bool = field(
            default=True, metadata={"help": "Whether to load model in 4 bit."}
        )

    @dataclass
    class gen_kwargs:
        max_new_tokens: int = field(default=50, metadata={"help": "Max new tokens."})
        do_sample: bool = field(default=True, metadata={"help": "Whether to sample."})
        temperature: float = field(default=1.0, metadata={"help": "Temperature."})


@dataclass
class TrainCfg:
    @dataclass
    class EarlyStopping:
        patience: int = field(
            default=10, metadata={"help": "Patience in training algorithm."}
        )
        delta: float = field(
            default=0.0, metadata={"help": "Delta in training algorithm."}
        )

    # Main configuration attributes
    seed: int = field(default=0, metadata={"help": "Random seed."})
    lr: float = field(
        default=0.0005,
        metadata={"help": "Learning rate to use for regression network."},
    )
    epochs: int = field(
        default=100,
        metadata={
            "help": "Epochs to train regression network for."
        },  # They did 500 each
    )
    seqs: int = field(
        default=60, metadata={"help": "Number of mini-batches to go through in total."}
    )
    alpha: float = field(
        default=0.05, metadata={"help": "Significance level."}
    )  # significance level
    T: int = field(default=0, metadata={"help": "Warm start."})
    batch_size: int = field(
        default=100, metadata={"help": "Batch size for sequences in test."}
    )  # 64 (they use 90)
    save_dir: Optional[str] = field(
        default="models", metadata={"help": "Directoy to save regression network to."}
    )
    save: Optional[bool] = field(
        default=True, metadata={"help": "Whether to save regression network."}
    )
    l1_lambda: float = field(default=0.0)
    l2_lambda: float = field(default=0.0)

    # Include the early_stopping configuration as a nested attribute
    earlystopping: EarlyStopping = field(default_factory=EarlyStopping)
    net_batch_size: int = field(
        default=100, metadata={"help": "Batch size of regression network."}
    )
