import argparse
import os
import sys

# imports from other scripts
from arguments import TrainCfg
from utils.utils import load_config

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep-anytime-testing")
)

from auditing_test.test import AuditingTest, CalibratedAuditingTest, eval_model


def main():
    """ """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--exp",
        type=str,
        choices=["generation", "test"],
        required=True,
        help="Select the experiment to run: generating model outputs or auditing test.",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to evaluate the model on the metric.",
    )

    # TODO: change this when OnlineTrainer is no longer deprecated.
    parser.add_argument(
        "--online",
        action="store_true",
        help="Whether to use the OnlineTrainer instead of the OfflineTrainer. Warning: OnlineTrainer is currently deprecated.",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yml",
        help="Path to config file",
    )

    parser.add_argument(
        "--fold_size",
        type=int,
        default=4000,
        help="Fold size when running kfold tests. Default is 4000.",
    )

    parser.add_argument(
        "--model_name1",
        type=str,
        default=None,
        help="Name of first model as it appears in the folder name.",
    )

    parser.add_argument(
        "--model_name2",
        type=str,
        default=None,
        help="Name of second model as it appears in the folder name.",
    )

    parser.add_argument(
        "--seed1",
        type=str,
        default=None,
        help="Generation seed of first model as it appears in the folder name.",
    )

    parser.add_argument(
        "--seed2",
        type=str,
        default=None,
        help="Generation seed of second model as it appears in the folder name.",
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="If this is set to true, then no tracking on wandb.",
    )

    parser.add_argument(
        "--no_analysis",
        action="store_true",
        help="If this is set to true, then no analysis after runnning the test.",
    )

    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="If this is set to true, then we run the test with calibrated epsilon",
    )

    args = parser.parse_args()
    config = load_config(args.config_path)

    # Determine which experiment to run based on the argument
    if args.exp == "generation":
        eval_model(config, evaluate=args.evaluate)

    elif args.exp == "test":
        train_cfg = TrainCfg()
        if args.calibrate:
            exp = CalibratedAuditingTest(
                config,
                train_cfg,
                use_wandb=not args.no_wandb,
            )
            exp.run(
                model_name1=args.model_name1,
                seed1=args.seed1,
                model_name2=args.model_name2,
                seed2=args.seed2,
                fold_size=args.fold_size,
            )
        else:
            exp = AuditingTest(
                config,
                train_cfg,
                use_wandb=not args.no_wandb,
            )
            exp.run(
                model_name1=args.model_name1,
                seed1=args.seed1,
                model_name2=args.model_name2,
                seed2=args.seed2,
                fold_size=args.fold_size,
                analyze_distance=not args.no_analysis,
            )


if __name__ == "__main__":
    config = load_config("config.yml")
    train_cfg = TrainCfg()
    model_name1 = "Meta-Llama-3-8B-Instruct"
    model_name2 = "Llama-3-8B-ckpt5"
    seed1 = "seed1000"
    seed2 = "seed1000"
    fold_size = 4000

    exp = CalibratedAuditingTest(config, train_cfg, use_wandb=False, bias=0)
    exp.run(
        model_name1=model_name1,
        seed1=seed1,
        model_name2=model_name2,
        seed2=seed2,
        fold_size=fold_size,
    )

    # exp = AuditingTest(config, train_cfg, use_wandb=False)
    # exp.run(
    #     model_name1=model_name1,
    #     seed1=seed1,
    #     model_name2=model_name2,
    #     seed2=seed2,
    #     fold_size=fold_size,
    # )

    # main()
