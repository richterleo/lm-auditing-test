# Auditing Test to Detect Behavioral Shift in Language Models

## Overview

[IN PROGRESS]

This repository accompanies the paper [An Auditing Test to Detect Behavioral Shift in Language Models](https://arxiv.org/abs/2410.19406) and provides tools to run two sets of experiments:


1. **Evaluating a Model with Respect to a Certain Behavior and Metric**: This involves evaluating a model, logging results to Weights and Biases (wandb), and generating plots.
2. **Detecting Change in Model Behavior using our Auditing Test**: This tests to distinguish between two distributions of model behavior based on individual paired samples.

## Setup

Ensure you have all the necessary dependencies installed. You can install miniconda and create a virtual environment `auditenv` with the necessary dependencies by executing 

```bash
source ./setup.sh
```
You will be prompted to enter your wandb API key and huggingface token.

The folder `configs` contains general configuration and specific configurations for experiments. 

## Evaluating a Model
The repository currently supports evaluating LLMs of the `Llama`, `Gemma`, `Mistral` and `aya`-families. When using LoRA, make sure to include the model in the list in `./configs/peft_models.yaml`. Behaviors supported are **toxicity** and **translation_performance** with different metrics each. Other models, behaviors and metrics can be added easily (see section ...)

Specify model, dataset and metric in `./configs/experiments/generation.yaml` and run

```bash
python main.py --exp generation
```
By default, generations will be saved locally and uploaded to wandb. 

## Auditing Test 
To run the Auditing test and compare two model distributions based on a behavior, use the following command:

```bash
python main.py --exp <test_config>
```
`<test_config>`s for toxicity and translation can be found in `./configs/experiments/`. Change models accordingly. 

## Configuration
The hyperparameters for the experiments are specified in the config files in `./configs` as well as in `arguments.py`. This file contains the training configuration settings.


## Logging and Plotting

Generation experiments store results on wandb by default. Add

```bash
python main.py --exp generation --no_wandb
```
to avoid tracking on wandb.

## Contact

For any questions or issues, please contact [the authors](leonie.richter.23@ucl.ac.uk).
