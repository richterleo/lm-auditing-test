# Distance Simulation

## Overview

This repository provides tools to run two sets of experiments:

1. **Evaluating a Model with Respect to a Certain Behavior and Metric**: This involves evaluating a model, logging results to Weights and Biases (wandb), and generating plots.
2. **Deep Anytime Testing (DAT)**: This tests to distinguish between two model distributions based on a behavior using the test described in this [paper](https://arxiv.org/abs/2310.19384)

## Setup

Ensure you have all the necessary dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```
or 
```bash
conda env create -f environment.yml
```

## Evaluating a Model
To run experiments that evaluate a model with respect to a certain behavior and metric, use the following command:

```bash
python main.py --exp evaluation
```
This will log results to wandb and generate plots based on the evaluation metrics.

## Deep Anytime Testing (DAT)
To run experiments that test the Deep Anytime Testing (DAT) to distinguish two model distributions based on a behavior, use the following command:

```bash
python main.py --exp test_dat
```

## Configuration
The hyperparameters for the experiments are specified in two files:

`config.yml`: This file contains the main configuration settings.
`arguments.py`: This file contains the training configuration settings as well as the model configuration settings

A second model configuration can be specified either directly in `config.yml` under `tau2` or be given to the function `test_dat()` in `main.py` as a `ModelCfg` imported from `arguments.py`.

## Logging and Plotting

The results of the evaluation experiments are logged to wandb. Please set your API key by: 

```
export WANDB_API_KEY= <your_api_key>
```
before running experiments with logging. 

## Contact

For any questions or issues, please contact [Your Name] at [richter.leo94@gmail.com].