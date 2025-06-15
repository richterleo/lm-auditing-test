import json
import logging
import wandb
import torch
import sys

from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import login
from omegaconf import OmegaConf
from os import getenv
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Subset
from transformers import (
    pipeline,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from transformers.utils import is_flash_attn_2_available
from typing import Optional, Dict, List
from peft import AutoPeftModelForCausalLM

# Add paths to sys.path if not already present
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lm_auditing.utils.utils import (
    translate_model_kwargs,
    NestedKeyDataset,
    terminator,
    format_funcs,
    check_seed,
    create_conversation,
    create_run_string,
    format_gen_params,
)

from lm_auditing.base.experiment_base import ExperimentBase

from lm_auditing.utils.logging_config import setup_logging

from configs.experiment_config import (
    GenerationConfig,
)

# hf_token = getenv("HF_TOKEN")
# login(hf_token, add_to_git_credential=False)


class ModelGenerator(ExperimentBase):
    def __init__(
        self,
        config: GenerationConfig,
    ):
        if is_flash_attn_2_available():
            config.model.model_kwargs.update({"attn_implementation": "flash_attention_2"})

        super().__init__(config)

    def _calculate_dependent_attributes(self):
        self.logging_cfg = self.config.logging
        self.model_cfg = self.config.model
        self.metric_cfg = self.config.metric
        self.eval_cfg = self.config.eval
        self.storing_cfg = self.config.storing

        dir_prefix = (
            self.storing_cfg.dir_prefix if self.storing_cfg.dir_prefix is not None else str(self.metric_cfg.metric)
        )
        self.output_dir = self.SCRIPT_DIR / dir_prefix / self.storing_cfg.output_dir

        self.model_id = f"{self.model_cfg.hf_prefix}/{self.model_cfg.model_id}"
        self.seed = check_seed(self.model_cfg.gen_seed)
        self.gen_kwargs = self.model_cfg.gen_kwargs.to_dict()
        self.include_gen_kwargs_in_path = not (self.model_cfg.gen_kwargs == self.model_cfg.default_gen_kwargs)

        self.model_kwargs = self.model_cfg.model_kwargs.to_dict()
        self.model_id = f"{self.model_cfg.hf_prefix}/{self.model_cfg.model_id}"

        self.setup_logger(self.model_cfg.model_id, self.seed, tag="generation")

    def generate(self, overrides: Optional[Dict] = None):
        if overrides:
            self._apply_overrides(overrides)

        try:
            self._generate_on_dataset()
        finally:
            if self.config.logging.use_wandb:
                wandb.finish()

    def _get_file_path(
        self,
        num_samples: int = -1,
        lower_index: Optional[int] = None,
        upper_index: Optional[int] = None,
    ) -> Path:
        """Creates and returns the file path for generation outputs, checking if it exists."""
        few_shot_string = "_fewshot" if self.metric_cfg.few_shot else ""
        # Create base filename
        file_name = "continuations"
        if num_samples == -1:
            file_name += ".json"
        elif upper_index and lower_index:
            file_name += f"_{lower_index}_{upper_index}.json"
        else:
            file_name += f"_{num_samples}.json"

        # Create folder path
        base_name = f"{self.model_id.split('/')[-1]}{few_shot_string}_seed{self.seed}"
        if self.include_gen_kwargs_in_path:
            gen_params_str = format_gen_params(self.gen_kwargs)
            base_name += f"_{gen_params_str}"

        folder_path = self.output_dir / base_name
        folder_path.mkdir(parents=True, exist_ok=True)

        return folder_path / file_name

    def _get_local_data_path(self, data_path) -> Path:
        few_shot_string = "_fewshot" if self.metric_cfg.few_shot else ""
        return data_path.parent / f"{data_path.stem}{few_shot_string}{data_path.suffix}"

    def _generate_on_dataset(self):
        torch.manual_seed(self.seed)
        ds_name = str(self.metric_cfg.dataset_name)

        # Get file path and check if generation is needed
        file_path = self._get_file_path(
            self.eval_cfg.num_samples,
            self.eval_cfg.part * 10000 if self.eval_cfg.eval_in_parts else None,
            (self.eval_cfg.part + 1) * 10000 if self.eval_cfg.eval_in_parts else None,
        )

        if file_path.exists() and not self.eval_cfg.overwrite:
            self.logger.info(f"File {file_path} already exists. Skipping inference")
            return

        # Setup tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")

        terminators = [tokenizer.eos_token_id]

        if "llama-3" in self.model_id.lower():
            terminators.append(tokenizer.convert_tokens_to_ids(terminator["llama3"]))
            format_func = format_funcs["llama3"](mode=self.model_cfg.chat_style)
        elif "mistral" in self.model_id.lower():
            terminators.append(tokenizer.convert_tokens_to_ids(terminator["mistral"]))
            format_func = format_funcs["mistral"](mode=self.model_cfg.chat_style)
        elif "gemma" in self.model_id.lower():
            terminators.append(tokenizer.convert_tokens_to_ids(terminator["gemma"]))
            format_func = format_funcs["gemma"](mode=self.model_cfg.chat_style)

        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # for dataset
        data_path = self.SCRIPT_DIR / ds_name
        local_data_path = self._get_local_data_path(data_path)

        if local_data_path.exists():
            self.logger.info(f"Loading dataset locally from {local_data_path}.")
            data_files = {self.metric_cfg.dataset_split: str(local_data_path.name)}
            prompt_dataset = load_dataset(
                local_data_path.parent.as_posix(),
                data_files=data_files,
            )[self.metric_cfg.dataset_split]
            ground_truths = [prompt_dataset[i]["output"] for i in range(len(prompt_dataset))]
            formatted_dataset = prompt_dataset.map(
                lambda x: create_conversation(x, self.model_id),
                remove_columns=prompt_dataset.features,
                batched=False,
                desc="Generating conversations for evaluation",
            )
            dataset = [formatted_dataset[i]["messages"] for i in range(len(formatted_dataset))]

        else:
            self.logger.info(f"Loading dataset {ds_name} from huggingface.")
            dataset = load_dataset(ds_name, split=self.metric_cfg.dataset_split)

        if self.model_cfg.use_peft:
            model = AutoPeftModelForCausalLM.from_pretrained(self.model_id, **self.model_kwargs)
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                model_kwargs=self.model_kwargs,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            # Use the model_kwargs from config which includes quantization and memory optimizations
            model_kwargs = translate_model_kwargs(self.model_kwargs)

            # Create quantization config from the model kwargs
            quantization_config = BitsAndBytesConfig(**model_kwargs.pop("quantization_config"))

            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=quantization_config, **model_kwargs
            )

            # Then create pipeline with the loaded model
            generator = pipeline(
                "text-generation",
                model=model,  # Use the loaded model directly
                tokenizer=tokenizer,
                pad_token_id=tokenizer.pad_token_id,
            )

        self.logger.info("Model loaded.")

        if self.eval_cfg.num_samples < len(dataset) and self.eval_cfg.num_samples != -1:
            if self.eval_cfg.sample_randomly:
                subset_indices = torch.randperm(len(dataset))[: self.eval_cfg.num_samples]
                dataset = Subset(dataset, subset_indices.tolist())
            else:
                dataset = Subset(
                    dataset,
                    range(self.eval_cfg.num_samples),
                )
        elif self.eval_cfg.eval_in_parts:
            lower_index = self.eval_cfg.part * self.eval_cfg.part_length
            upper_index = (self.eval_cfg.part + 1) * self.eval_cfg.part_length

            if lower_index >= len(dataset):
                raise ValueError(f"Lower index {lower_index} is out of range for dataset of length {len(dataset)}")

            if upper_index > len(dataset):
                self.logger.info(
                    f"Upper index {upper_index} is larger than dataset length {len(dataset)}. Will use dataset length instead."
                )

            dataset = Subset(
                dataset,
                range(
                    lower_index,
                    min(upper_index, len(dataset)),
                ),
            )

        logs = defaultdict(list)
        logs["metadata"] = {
            "dataset_name": ds_name,
            "model_id": self.model_id,
            "gen_kwargs": {k: str(v) for k, v in self.gen_kwargs.items()},
            "num_samples": self.eval_cfg.num_samples,
            "batch_size": self.eval_cfg.batch_size,
            "seed": self.seed,
            "use_wandb": self.logging_cfg.use_wandb,
            "behavior": str(self.metric_cfg.behavior),
            "metric": str(self.metric_cfg.metric),
            "few_shot": self.metric_cfg.few_shot,
        }

        def _save_logs(current_logs, sample_count):
            """Helper function to save logs at a given sample count"""
            interim_file_name = f"continuations_{sample_count}.json"
            interim_file_path = file_path.parent / interim_file_name
            with open(interim_file_path, "w") as file:
                json.dump(
                    current_logs,
                    file,
                    ensure_ascii=False,
                    indent=4,
                )
            if self.logging_cfg.use_wandb:
                wandb.save(str(interim_file_path))
            self.logger.info(f"Saved intermediate results at {sample_count} samples.")

        # bit hacky, but for some reason with translation dataset, we need to feed prompts individually or else it takes too long
        if local_data_path.exists():
            # for translation case, we have ground truths and continuations
            for i, input in enumerate(tqdm(dataset)):
                out = generator(
                    [input],
                    eos_token_id=terminators,
                    return_full_text=False,
                    **self.gen_kwargs,
                )
                logs["continuations"].append(out[0][0]["generated_text"])
                logs["ground_truths"].append(ground_truths[i])

                if self.eval_cfg.save_intermittently and (i + 1) % self.eval_cfg.save_interval == 0:
                    _save_logs(logs, i + 1)

        else:
            # in toxicity case, we have continuations and prompts
            for i, out in tqdm(
                enumerate(
                    generator(
                        NestedKeyDataset(
                            dataset,
                            "prompt",
                            "text",
                            self.model_id,
                            format_func,
                            tokenizer,
                        ),
                        batch_size=self.eval_cfg.batch_size,
                        eos_token_id=terminators,
                        return_full_text=False,
                        **self.gen_kwargs,
                    )
                ),
                total=len(dataset),
            ):
                logs["prompts"].append(dataset[i]["prompt"]["text"])
                logs["continuations"].append(out[0]["generated_text"])

                if self.eval_cfg.save_intermittently and (i + 1) % self.eval_cfg.save_interval == 0:
                    _save_logs(logs, i + 1)

        # Save final results
        with open(file_path, "w") as file:
            json.dump(logs, file, ensure_ascii=False, indent=4)

        if self.logging_cfg.use_wandb:
            wandb.save(file_path)
