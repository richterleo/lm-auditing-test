import torch

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Literal, Union


# Logging Configuration
@dataclass
class LoggingConfig:
    use_wandb: bool
    entity: str
    quiet: bool
    wandb_project_name: str


@dataclass
class StoringConfig:
    dir_prefix: Optional[str]
    output_dir: str
    score_dir: str
    test_dir: str
    plot_dir: str


# Network Configuration
@dataclass
class NetworkConfig:
    input_size: int
    hidden_layer_size: List[int]
    layer_norm: bool
    bias: bool
    model_type: str  # "CMLP" or "HighCapacityCMLP"
    num_layers: int
    batch_size: int
    drop_out: bool
    drop_out_p: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Training Configuration
@dataclass
class EarlyStoppingConfig:
    patience: int
    delta: float


@dataclass
class CalibrationConfig:
    epsilon_strategy: str
    epsilon_ticks: int
    lower_interval_end: float
    upper_interval_end: float
    lower_model_name: str
    lower_model_seed: str
    upper_model_name: str
    upper_model_seed: str
    num_runs: int


@dataclass
class AnalysisConfig:
    calculate_distance: bool
    unpaired: bool
    num_runs: int
    num_samples: int
    multiples_of_epsilon: int
    use_full_ds_for_nn_distance: bool
    epsilon_ticks: int


@dataclass
class TrainingConfig:
    earlystopping: EarlyStoppingConfig
    seed: int
    lr: float
    epochs: int
    seqs: int
    alpha: float
    T: int
    batch_size: int
    save_dir: str
    save: bool
    l1_lambda: float
    l2_lambda: float
    net_batch_size: int

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "TrainingConfig":
        early_stopping_dict = cfg_dict.pop("earlystopping")
        return cls(
            earlystopping=EarlyStoppingConfig(**early_stopping_dict),
            **cfg_dict,
        )


# Evaluation Configuration
@dataclass
class EvalConfig:
    num_samples: int
    batch_size: int
    use_vllm: bool
    overwrite: bool
    eval_in_parts: bool
    part: int
    save_intermittently: bool
    save_interval: int
    sample_randomly: bool
    part_length: int


# Model Generation Configuration
@dataclass
class GenerationKwargs:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "GenerationKwargs":
        return cls(**cfg_dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuantizationConfig:
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str
    bnb_4bit_use_double_quant: bool

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "QuantizationConfig":
        return cls(**cfg_dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelKwargs:
    quantization_config: QuantizationConfig
    low_cpu_mem_usage: bool
    device_map: Optional[str]
    torch_dtype: Optional[torch.dtype] = None
    attn_implementation: Optional[str] = None
    max_memory: Optional[Dict] = None
    offload_folder: Optional[str] = None

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "ModelKwargs":
        # Handle nested quantization config
        quant_config = QuantizationConfig.from_dict(cfg_dict.pop("quantization_config"))

        # Handle torch dtype conversion
        if "torch_dtype" in cfg_dict:
            dtype_str = cfg_dict["torch_dtype"]
            dtype_map = {
                "torch.float16": torch.float16,
                "torch.float32": torch.float32,
                "torch.bfloat16": torch.bfloat16,
            }
            cfg_dict["torch_dtype"] = dtype_map.get(dtype_str)

        return cls(quantization_config=quant_config, **cfg_dict)

    def to_dict(self) -> Dict[str, Any]:
        # Start with the base dictionary from asdict
        result = asdict(self)

        # Handle torch_dtype conversion to string
        if self.torch_dtype is not None:
            dtype_map = {
                torch.float16: "torch.float16",
                torch.float32: "torch.float32",
                torch.bfloat16: "torch.bfloat16",
            }
            result["torch_dtype"] = dtype_map.get(self.torch_dtype)

        # Convert nested QuantizationConfig to dict
        result["quantization_config"] = self.quantization_config.to_dict()

        return result


@dataclass
class ModelConfig:
    model_id: str
    gen_seed: str
    model_kwargs: ModelKwargs
    gen_kwargs: GenerationKwargs  # GenerationConfig
    default_gen_kwargs: GenerationKwargs  #  GenerationConfig
    use_peft: Optional[bool] = None
    chat_style: str = "default"
    hf_prefix: Optional[str] = None

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "ModelConfig":
        return cls(
            model_id=cfg_dict["model_id"],
            gen_seed=cfg_dict["gen_seed"],
            model_kwargs=ModelKwargs.from_dict(cfg_dict["model_kwargs"]),
            gen_kwargs=GenerationKwargs.from_dict(cfg_dict["gen_kwargs"]),
            default_gen_kwargs=GenerationKwargs.from_dict(cfg_dict["default_gen_kwargs"]),
            use_peft=cfg_dict.get("use_peft"),
            chat_style=cfg_dict.get("chat_style"),
            hf_prefix=cfg_dict.get("hf_prefix"),
        )


# Metric Configuration
@dataclass
class MetricConfig:
    behavior: str
    metric: str
    lower_lim: float
    upper_lim: float
    dataset_name: str
    dataset_split: str
    few_shot: bool


# Test Configuration
@dataclass
class TestParams:
    calibrate: bool
    analyze_distance: bool
    only_continuations: bool
    fold_size: int
    overwrite: bool
    noise: float
    track_c2st: bool
    calibrate_only: bool
    calibration_params: CalibrationConfig
    analysis_params: AnalysisConfig
    epsilon: float
    run_davtt: bool

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "TestParams":
        calibration_params_dict = cfg_dict.pop("calibration_params")
        analysis_params_dict = cfg_dict.pop("analysis_params")
        return cls(
            calibration_params=CalibrationConfig(**calibration_params_dict),
            analysis_params=AnalysisConfig(**analysis_params_dict),
            **cfg_dict,
        )


@dataclass
class BaseConfig:
    logging: LoggingConfig
    metric: MetricConfig
    storing: StoringConfig
    exp: Literal["generation", "test"]


@dataclass
class GenerationConfig(BaseConfig):
    model: ModelConfig
    eval: EvalConfig

    def to_dict(self, for_wandb: bool = False) -> Dict[str, Any]:
        """Convert the config to a dictionary, with special handling for wandb serialization.

        Args:
            for_wandb (bool): If True, converts torch datatypes to strings for wandb compatibility
        """
        # Start with base asdict conversion
        result = {
            "logging": asdict(self.logging),
            "metric": asdict(self.metric),
            "storing": asdict(self.storing),
            "eval": asdict(self.eval),
            "exp": self.exp,
        }

        # Handle ModelConfig which contains nested ModelKwargs
        model_dict = {
            "model_id": self.model.model_id,
            "gen_seed": self.model.gen_seed,
            "use_peft": self.model.use_peft,
            "chat_style": self.model.chat_style,
            "hf_prefix": self.model.hf_prefix,
        }

        # Handle generation kwargs
        model_dict["gen_kwargs"] = self.model.gen_kwargs.to_dict()
        model_dict["default_gen_kwargs"] = self.model.default_gen_kwargs.to_dict()

        # Handle model kwargs with torch dtype conversion
        model_kwargs = self.model.model_kwargs.to_dict()
        if for_wandb and "torch_dtype" in model_kwargs:
            # Convert torch dtype to string for wandb
            dtype_map = {
                torch.float16: "torch.float16",
                torch.float32: "torch.float32",
                torch.bfloat16: "torch.bfloat16",
            }
            if model_kwargs["torch_dtype"] in dtype_map:
                model_kwargs["torch_dtype"] = dtype_map[model_kwargs["torch_dtype"]]

        model_dict["model_kwargs"] = model_kwargs
        result["model"] = model_dict

        return result


@dataclass
class TestConfig(BaseConfig):
    model1: ModelConfig
    model2: ModelConfig
    net: NetworkConfig
    training: TrainingConfig
    test_params: TestParams

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "TestConfig":
        test_params_dict = cfg_dict.pop("test_params")
        training_dict = cfg_dict.pop("training")

        test_params = TestParams(**test_params_dict)
        training = TrainingConfig(**training_dict)

        return cls(
            logging=LoggingConfig(**cfg_dict.pop("logging")),
            metric=MetricConfig(**cfg_dict.pop("metric")),
            storing=StoringConfig(**cfg_dict.pop("storing")),
            model1=ModelConfig(**cfg_dict.pop("model1")),
            model2=ModelConfig(**cfg_dict.pop("model2")),
            net=NetworkConfig(**cfg_dict.pop("net")),
            test_params=test_params,
            training=training,
            exp=cfg_dict.pop("exp"),
        )


ExperimentConfig = Union[GenerationConfig, TestConfig]
