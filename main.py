import hydra
import logging
import os
import sys

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional, Dict, List

# Add paths to sys.path if not already present
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# imports from other scripts
from arguments import TrainCfg

# from logging_config import setup_logging
from src.utils.utils import load_config

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

SCRIPT_DIR = SCRIPT_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

# def main():
#     """ """
#     # Create the argument parser
#     parser = argparse.ArgumentParser(description="Run experiments")
#     parser.add_argument(
#         "--exp",
#         type=str,
#         choices=["generation", "test"],
#         required=True,
#         help="Select the experiment to run: generating model outputs or auditing test.",
#     )

#     # TODO: change this when OnlineTrainer is no longer deprecated.
#     parser.add_argument(
#         "--online",
#         action="store_true",
#         help="Whether to use the OnlineTrainer instead of the OfflineTrainer. Warning: OnlineTrainer is currently deprecated.",
#     )

#     parser.add_argument(
#         "--config_path",
#         type=str,
#         default="config.yml",
#         help="Path to config file",
#     )

#     parser.add_argument(
#         "--fold_size",
#         type=int,
#         default=4000,
#         help="Fold size when running kfold tests. Default is 4000.",
#     )

#     parser.add_argument(
#         "--model_name1",
#         type=str,
#         default=None,
#         help="Name of first model as it appears in the folder name.",
#     )

#     parser.add_argument(
#         "--model_name2",
#         type=str,
#         default=None,
#         help="Name of second model as it appears in the folder name.",
#     )

#     parser.add_argument(
#         "--seed1",
#         type=str,
#         default=None,
#         help="Generation seed of first model as it appears in the folder name.",
#     )

#     parser.add_argument(
#         "--seed2",
#         type=str,
#         default=None,
#         help="Generation seed of second model as it appears in the folder name.",
#     )

#     parser.add_argument(
#         "--no_wandb",
#         action="store_true",
#         help="If this is set to true, then no tracking on wandb.",
#     )

#     parser.add_argument(
#         "--no_analysis",
#         action="store_true",
#         help="If this is set to true, then no analysis after runnning the test.",
#     )

#     parser.add_argument(
#         "--calibrate",
#         action="store_true",
#         help="If this is set to true, then we run the test with calibrated epsilon",
#     )

#     parser.add_argument("--debug_mode", action="store_true", help="Run in debug mode")

#     parser.add_argument("--high_temp", action="store_true", help="Run with high temperature")

#     parser.add_argument("--hf_prefix", type=str, default=None, help="Prefix for huggingface model")

#     parser.add_argument("--eval_on_task", action="store_true", help="Whether to evaluate on task")

#     parser.add_argument("--few_shot", action="store_true", help="Whether to run few shot experiments")

#     parser.add_argument(
#         "--test_on",
#         default="toxicity",
#         help="Whether to test on on toxicity dataset or translation dataset or waterbirds.",
#     )

#     args = parser.parse_args()

#     if args.debug_mode:
#         debugpy.listen(("0.0.0.0", 5678))
#         print("waiting for debugger attach...")
#         debugpy.wait_for_client()
#         print("Debugger attached")

#     config = load_config(args.config_path)
#     if args.hf_prefix:
#         config["tau1"]["hf_prefix"] = args.hf_prefix

#     if args.few_shot:
#         config["task_metric"]["few_shot"] = True
#     else:
#         config["task_metric"]["few_shot"] = False

#     if args.test_on_task:
#         config["metric"] = config["task_metric"]

#     # Determine which experiment to run based on the argument
#     if args.exp == "generation":
#         # TODO: make this a bit smoother
#         if args.high_temp:
#             config["tau1"]["gen_kwargs"] = config["tau1"]["gen_kwargs_high_temp"]
#         eval_model(
#             config,
#             model_id=args.model_name1,
#             hf_prefix=args.hf_prefix,
#             use_wandb=not args.no_wandb,
#             eval_on_task=args.eval_on_task,
#         )

#     elif args.exp == "test":
#         train_cfg = TrainCfg()
#         if args.calibrate:
#             exp = CalibratedAuditingTest(
#                 config,
#                 train_cfg,
#                 DefaultEpsilonStrategy(config),
#                 use_wandb=not args.no_wandb,
#             )
#             exp.run(
#                 model_name1=args.model_name1,
#                 seed1=args.seed1,
#                 model_name2=args.model_name2,
#                 seed2=args.seed2,
#                 fold_size=args.fold_size,
#             )
#         else:
#             if args.test_on_task:
#                 exp = AuditingTest(
#                     config,
#                     train_cfg,
#                     use_wandb=not args.no_wandb,
#                     only_continuations=False,
#                     output_dir="processed_data/translation_test_outputs",
#                     test_on_task=True,
#                 )
#                 exp.run(
#                     model_name1=args.model_name1,
#                     seed1=args.seed1,
#                     model_name2=args.model_name2,
#                     seed2=args.seed2,
#                     fold_size=args.fold_size,
#                     analyze_distance=not args.no_analysis,
#                 )

#             elif args.test_on_waterbirds:
#                 exp = AuditingTest(
#                     config,
#                     train_cfg,
#                     use_wandb=not args.no_wandb,
#                     only_continuations=False,
#                     output_dir="BalancingGroups/outputs",
#                     test_on_waterbirds=True,
#                 )
#                 exp.run(
#                     model_name1=args.model_name1,
#                     seed1=args.seed1,
#                     model_name2=args.model_name2,
#                     seed2=args.seed2,
#                     fold_size=args.fold_size,
#                     analyze_distance=not args.no_analysis,
#                 )
#             else:
#                 exp = AuditingTest(
#                     config,
#                     train_cfg,
#                     use_wandb=not args.no_wandb,
#                 )
#                 exp.run(
#                     model_name1=args.model_name1,
#                     seed1=args.seed1,
#                     model_name2=args.model_name2,
#                     seed2=args.seed2,
#                     fold_size=args.fold_size,
#                     analyze_distance=not args.no_analysis,
#                 )


# if __name__ == "__main__":
#     config = load_config("/root/Auditing_test_for_LMs/Auditing_test_for_LMs/Auditing_test_for_LMs/config.yml")
#     train_cfg = TrainCfg()
#     output_dir = "BalancingGroups/outputs"
#     # model_name1 = "Meta-Llama-3-8B-Instruct"

#     # model_name_checkpoint = "Llama-3-8B-ckpt5"
#     # multiples_of_epsilon = 5

#     # model_name2 = "1-Meta-Llama-3-8B-Instruct"
#     # model_name3 = "2-Meta-Llama-3-8B-Instruct"
#     # model_name4 = "3-Meta-Llama-3-8B-Instruct"
#     # model_name5 = "4-Meta-Llama-3-8B-Instruct"
#     # model_name6 = "5-Meta-Llama-3-8B-Instruct"

#     # lower_model = "Meta-Llama-3-8B-Instruct-hightemp"
#     # lower_seed = "seed1000"

#     # upper_model = "Llama-3-8B-ckpt1"
#     # upper_seed = "seed2000"

#     model_name1 = "tr"
#     seed1 = "seed1000"
#     model_name2 = "te"
#     seed2 = "seed1000"

#     # upper_model = "LLama-3-8b-Uncensored"
#     # upper_seed = "seed1000"

#     # seed1 = "seed1000"
#     # seed2 = "seed1000"
#     fold_size = 4000

#     # tasks = [
#     #     "Mistral-7B-Instruct-v0.2",
#     #     "gemma-1.1-7b-it",
#     #     "Llama-3-8B-ckpt10",
#     #     "codealpaca-Meta-Llama-3-8B-Instruct",
#     #     "Meta-Llama-3-8B-Instruct-hightemp",
#     #     "commonsense_classification-Meta-Llama-3-8B-Instruct",
#     #     "program_execution-Meta-Llama-3-8B-Instruct",
#     #     "sentence_perturbation-Meta-Llama-3-8B-Instruct",
#     #     "text_matching-Meta-Llama-3-8B-Instruct",
#     #     "textual_entailment-Meta-Llama-3-8B-Instruct",
#     # ]

#     # task_seeds = ["seed2000"] + ["seed1000" for _ in range(len(tasks) - 1)]

#     # models_and_seeds = [
#     #     {"model_name": task, "seed": seed} for task, seed in zip(tasks, task_seeds)
#     # ]

#     # exp = CalibratedAuditingTest(
#     #     config,
#     #     train_cfg,
#     #     IntervalEpsilonStrategy(lower_model, lower_seed, upper_model, upper_seed, config=config, num_runs=20),
#     #     use_wandb=False,
#     #     overwrite=False,
#     # )

#     # exp.run(model_name1=model_name1, seed1="seed1000", model_name2=model_name2, seed2="seed1000", fold_size=fold_size)

#     # exp.run(model_name1=model_name1, seed1="seed1000", model_name2=model_name3, seed2="seed1000", fold_size=fold_size)

#     # exp.run(model_name1=model_name1, seed1="seed1000", model_name2=model_name4, seed2="seed1000", fold_size=fold_size)

#     # exp.run(model_name1=model_name1, seed1="seed1000", model_name2=model_name5, seed2="seed1000", fold_size=fold_size)

#     # exp.run(model_name1=model_name1, seed1="seed1000", model_name2=model_name6, seed2="seed1000", fold_size=fold_size)

#     # exp = CalibratedAuditingTest(
#     #     config,
#     #     train_cfg,
#     #     DefaultEpsilonStrategy(config=config, num_runs=20),
#     #     use_wandb=False,
#     #     only_continuations=True,
#     # )
#     # exp.run(
#     #     model_name1=model_name1,
#     #     seed1="seed1000",
#     #     model_name2=model_name_checkpoint,
#     #     seed2="seed1000",
#     #     fold_size=fold_size,
#     # )

#     # exp = AuditingTest(config, train_cfg, use_wandb=False)
#     # exp.run(
#     #     model_name1=model_name1,
#     #     seed1=seed1,
#     #     model_name2=model_name2,
#     #     seed2=seed2,
#     #     fold_size=fold_size,
#     # )

#     # eval_model(config, use_wandb=False, eval_on_task=True)
#     # main()

#     # import json

#     # with open(
#     #     "/root/Auditing_test_for_LMs/Auditing_test_for_LMs/processed_data/translation_model_scores/Meta-Llama-3-8B-Instruct_few_shot_seed1000/bleu_scores.json",
#     #     "r",
#     # ) as file:
#     #     data = json.load(file)

#     # print(len(data["bleu_scores"]))

#     exp = AuditingTest(
#         config,
#         train_cfg,
#         use_wandb=False,
#         only_continuations=True,
#         output_dir=output_dir,
#         test_on_waterbirds=True,
#     )

#     exp.run(model_name1=model_name1, seed1=seed1, model_name2=model_name2, seed2=seed2, fold_size=fold_size)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # If debug mode is enabled
    if cfg.get("debug_mode", False):
        import debugpy

        debugpy.listen(("0.0.0.0", 5678))
        logger.info("waiting for debugger attach...")
        debugpy.wait_for_client()
        logger.info("Debugger attached")

    # Convert configuration to a dictionary if needed
    # cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Determine which experiment to run based on cfg.exp
    if cfg.exp == "generation":
        # Instantiate and run the GenerationExperiment
        from src.test.experiments import GenerationExperiment

        experiment = GenerationExperiment(cfg)
        experiment.run()

    elif cfg.exp == "test":
        # Instantiate and run the appropriate TestExperiment
        from src.test.experiments import TestExperiment

        train_cfg = TrainCfg()
        experiment = TestExperiment(cfg, train_cfg)
        experiment.run()

    #     if cfg.get("calibrate", False):
    #         exp = CalibratedAuditingTest(
    #             cfg,
    #             train_cfg,
    #             DefaultEpsilonStrategy(cfg),
    #             use_wandb=cfg.logging.use_wandb,
    #         )
    #     else:
    #         exp = AuditingTest(
    #             cfg,
    #             train_cfg,
    #             use_wandb=cfg.logging.use_wandb,
    #             only_continuations=False,
    #             output_dir=cfg.get("output_dir", "default_output_dir"),
    #             test_on_task=cfg.get("test_on_task", False),
    #             test_on_waterbirds=cfg.get("test_on_waterbirds", False),
    #         )
    #     exp.run(
    #         model_name1=cfg.tau1.model_id,
    #         seed1=cfg.tau1.gen_seed,
    #         model_name2=cfg.get("tau2", {}).get("model_id"),
    #         seed2=cfg.get("tau2", {}).get("gen_seed"),
    #         fold_size=cfg.get("fold_size", 4000),
    #         analyze_distance=cfg.get("analysis", {}).get("calculate_distance", True),
    #     )
    # else:
    #     raise ValueError(f"Unknown experiment type: {cfg.exp}")


if __name__ == "__main__":
    main()
