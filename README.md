# Auditing Test to Detect Behavioral Shift in Language Models

## Overview

[IN PROGRESS]

This repository accompanies the paper [An Auditing Test to Detect Behavioral Shift in Language Models](https://arxiv.org/abs/2410.19406) and provides tools to run two sets of experiments:


1. **Evaluating a Model with Respect to a Certain Behavior and Metric**: This involves evaluating a model, logging results to Weights and Biases (wandb), and generating plots.
2. **Detecting Change in Model Behavior using our Auditing Test**: This tests to distinguish between two distributions of model behavior based on individual paired samples.

## Quick Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Configure API keys**:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your actual API keys:
   - **WANDB_API_KEY**: Get from [Weights & Biases Settings](https://wandb.ai/settings)
   - **HF_TOKEN**: Get from [Hugging Face Tokens](https://huggingface.co/settings/tokens)
   - **PERSPECTIVE_API_KEY**: Get from [Perspective API](https://developers.perspectiveapi.com/s/docs-get-started)

4. **Run the project**:
   ```bash
   uv run python main.py
   ```

## Memory Optimization

If you encounter "No space left on device" errors or memory issues:

1. **Clean HuggingFace cache**:
   ```bash
   python clear_hf_cache.py
   ```

2. **Set custom cache directories** (in your `.env` file):
   ```bash
   TRANSFORMERS_CACHE=/path/to/larger/disk/transformers_cache
   HF_HOME=/path/to/larger/disk/hf_home
   ```

3. **Memory optimizations** are already configured in `configs/config.yaml`:
   - 4-bit quantization with bfloat16
   - Automatic device mapping (`device_map: auto`)
   - Low CPU memory usage during loading

## Setup

This project uses `uv` to manage dependencies. Install `uv` by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/). Then clone the repository and initialize the submodule:

```bash
git clone --recursive https://github.com/richterleo/lm-auditing-test.git
```
Alternatively, clone the repository and initialize the submodule separately:

```bash
git clone https://github.com/richterleo/lm-auditing-test.git
git submodule init
git submodule update
```
The project uses [deep-anytime-testing](https://github.com/tpandeva/deep-anytime-testing) as a git submodule. 
The specific version of deep-anytime-testing is can be found in `.gitmodules`.

Install the dependencies by running:
```bash
uv sync
```




```bash
./setup.sh
```
(or `source ./setup.sh` if using tmux). You will be prompted to enter your wandb API key and huggingface token.

The folder `configs` contains general configuration and specific configurations for experiments. 

## Evaluating a Model
The repository currently supports evaluating LLMs of the `Llama`, `Gemma`, `Mistral` and `aya`-families. When using LoRA, make sure to include the model in the list in `./configs/peft_models.yaml`. 

Behaviors supported are **toxicity** and **translation_performance** with different metrics each. Other models, behaviors and metrics can be added easily (see section ...)

When evaluating translation performance, execute 

```bash
git clone https://github.com/allenai/natural-instructions.git
python ./src/utils/preprocessing_superni.py
```

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

## Dependencies

This project uses [deep-anytime-testing](https://github.com/tpandeva/deep-anytime-testing) as a git submodule. After cloning this repository, initialize and update the submodule:

```bash
git submodule init
git submodule update
```
The specific version of deep-anytime-testing is can be found in `.gitmodules`.

## Contact

For any questions or issues, please contact [the authors](leonie.richter.23@ucl.ac.uk).
