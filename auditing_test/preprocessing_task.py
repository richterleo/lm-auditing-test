import os
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
import wandb
import sys

from evaluate import load
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Tuple, Dict


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import create_run_string

logger = logging.getLogger(name=__file__)
logger.setLevel(logging.INFO)

# Configure file handler
file_handler = logging.FileHandler("preprocessing.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

base_dir = Path(__file__).parent.absolute()


def create_prompt(task_data, instance_input):
    definition = task_data.get("Definition", [""])[0]

    positive_examples = task_data.get("Positive Examples", [])
    negative_examples = task_data.get("Negative Examples", [])

    if not (positive_examples or negative_examples):
        return None

    prompt_parts = [f"Definition: {definition}\n"]

    for i, example in enumerate(positive_examples, start=1):
        prompt_parts.extend(
            [
                f"Positive Example {i}—",
                f"input: {example.get('input', '')}",
                f"output: {example.get('output', '')}",
                f"explanation: {example.get('explanation', '')}\n",
            ]
        )

    for i, example in enumerate(negative_examples, start=1):
        prompt_parts.extend(
            [
                f"Negative Example {i}—",
                f"input: {example.get('input', '')}",
                f"output: {example.get('output', '')}",
                f"explanation: {example.get('explanation', '')}\n",
            ]
        )

    prompt_parts.extend(["Now complete the following example—", f"input: {instance_input}", f"output:"])

    return "\n".join(prompt_parts)


def process_task(
    task="translation",
    data_path="natural-instructions/tasks",
    out_path="processed_data",
    save_prompt_lengths=True,
    overwrite=False,
    task_file_list=None,
    save_prompts=True,
    thresh_instance=1800,
    thresh_prompt=2200,
    verbose=True,
):
    base_dir = Path.cwd()
    data_path = Path(base_dir) / Path(data_path)
    out_path = Path(base_dir) / Path(out_path)
    category_path = out_path / task
    os.makedirs(category_path, exist_ok=True)
    prompt_length_dict = {"bare_data": [], "few_shot_data": []}

    skipped_long_prompts = 0

    if verbose:
        logger.info(f"Task file list: {task_file_list}")

    if task == "translation":
        language_dict = {}
        bare_prompts = os.path.join(category_path, f"{task}_data.jsonl")
        few_shot_prompts = os.path.join(category_path, f"{task}_data_few_shot.jsonl")

        if os.path.exists(bare_prompts) and os.path.exists(few_shot_prompts) and not overwrite:
            if verbose:
                logger.info(f"{task} data already exists")
        else:
            bare_data = []
            few_shot_data = []

            for file_name in os.listdir(data_path):
                if file_name.endswith(".json"):
                    file_path = os.path.join(data_path, file_name)
                    with open(file_path, "r") as json_file:
                        try:
                            data = json.load(json_file)

                            # Check if task_file_list is provided and if the current file is in the list
                            if task_file_list is not None:
                                if file_name not in task_file_list:
                                    continue
                            else:
                                # If task_file_list is not provided, use the original category check
                                current_categories = data.get("Categories", [])
                                current_categories = [cat.lower().replace(" ", "_") for cat in current_categories]
                                if task not in current_categories:
                                    continue

                            if task == "translation":
                                language_dict[file_name] = {
                                    "input_language": data.get("Input_language", ""),
                                    "output_language": data.get("Output_language", ""),
                                }

                            if verbose:
                                logger.info(f"Processing file: {file_name}")
                            instruction = data.get("Definition", [None])[0]
                            instances = data.get("Instances", [])

                            for instance in instances:
                                instance_input = instance.get("input", "")
                                instance_outputs = instance.get("output", [])
                                prompt = create_prompt(data, instance_input)
                                if not prompt:
                                    # this means that there are no examples for few shot prompts.
                                    continue
                                output = random.choice(instance_outputs)

                                if len(instance_input) > thresh_instance or len(prompt) > thresh_prompt:
                                    if verbose:
                                        logger.info(f"Long prompt skipped")
                                    skipped_long_prompts += 1
                                    continue
                                bare_data.append(
                                    {"instruction": instruction, "input": instance_input, "output": output}
                                )
                                few_shot_data.append({"prompt": prompt, "output": output})
                                prompt_length_dict["bare_data"].append(len(instance_input))
                                prompt_length_dict["few_shot_data"].append(len(prompt))

                        except json.JSONDecodeError:
                            logger.info(f"Error decoding JSON from file: {file_name}")

            if save_prompt_lengths:
                for key in prompt_length_dict:
                    avg_length = sum(prompt_length_dict[key]) / len(prompt_length_dict[key])
                    if verbose:
                        logger.info(f"Average prompt length for {key}: {avg_length}")
            if verbose:
                logger.info(f"Creating files for {task} data...")

            if save_prompts:
                with open(bare_prompts, "w") as f:
                    for entry in bare_data:
                        f.write(json.dumps(entry) + "\n")

                with open(few_shot_prompts, "w") as f:
                    for entry in few_shot_data:
                        f.write(json.dumps(entry) + "\n")

            if task == "translation":
                with open(os.path.join(category_path, "language_dict.json"), "w") as f:
                    json.dump(language_dict, f)

            if verbose:
                logger.info(f"Files saved to {bare_prompts} and {few_shot_prompts}")


def analyze_long_prompts(file_path):
    def read_jsonl(file_path: str) -> List[Dict[str, str]]:
        with open(file_path, "r", encoding="utf-8") as file:
            return [json.loads(line) for line in file]

    def get_prompt_length(item: Dict[str, str]) -> int:
        if "prompt" in item:
            return len(item["prompt"])
        elif "instruction" in item and "input" in item:
            return len(item["instruction"] + item["input"])
        elif "instruction" in item:
            return len(item["instruction"])
        else:
            raise ValueError("Unexpected data format")

    # Read the JSONL file
    data = read_jsonl(file_path)

    # Calculate prompt lengths
    prompt_lengths = [get_prompt_length(item) for item in data]

    # Calculate statistics
    max_length = max(prompt_lengths)
    max_index = prompt_lengths.index(max_length)
    avg_length = sum(prompt_lengths) / len(prompt_lengths)

    # Create a plot
    plt.figure(figsize=(12, 6))
    plt.plot(prompt_lengths, marker="o", markersize=3, linestyle="-", linewidth=1)
    plt.title("Prompt Lengths Over Dataset")
    plt.xlabel("Item Index")
    plt.ylabel("Prompt Length (characters)")

    # Highlight the maximum point
    plt.plot(max_index, max_length, "ro", markersize=10, label=f"Max: {max_length} at index {max_index}")
    plt.legend()

    # Add grid lines
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save the plot
    plt.savefig(f"prompt_lengths_{str(Path(file_path).name)}.png")
    plt.close()

    # Print results
    print(f"Average prompt length: {avg_length:.2f} characters")
    print(f"Maximum prompt length: {max_length} characters at index {max_index}")
    print(f"Plot saved as prompt_lengths_{str(Path(file_path).name)}.png")


def get_languages(language_file="processed_data/translation/language_dict.json"):
    with open(language_file, "r") as f:
        data = json.load(f)

    # Create a dictionary to store language pairs and their corresponding files
    language_pairs = {}

    # Process each file in the JSON data
    for filename, info in data.items():
        input_lang = info["input_language"][0]
        output_lang = info["output_language"][0]
        pair = (input_lang, output_lang)

        if pair not in language_pairs:
            language_pairs[pair] = []
        language_pairs[pair].append(filename)

    # Save the results as a JSONL file
    with open("processed_data/translation/language_pairs_output.jsonl", "w") as outfile:
        for pair, files in language_pairs.items():
            json_object = {"input_language": pair[0], "output_language": pair[1], "files": files}
            json.dump(json_object, outfile)
            outfile.write("\n")


def is_english_task(entry):
    return entry["input_language"] == "English" or entry["output_language"] == "English"


def is_english_french_task(entry):
    return (entry["input_language"] == "English" and entry["output_language"] == "French") or (
        entry["input_language"] == "French" and entry["output_language"] == "English"
    )


def get_english_french_tasks(language_pairs_file="processed_data/translation/language_pairs_output"):
    english_french_tasks = []

    # Read the input file and filter tasks
    with open(language_pairs_file, "r") as infile:
        for line in infile:
            entry = json.loads(line)
            if is_english_french_task(entry):
                english_french_tasks.append(
                    {
                        "input_language": entry["input_language"],
                        "output_language": entry["output_language"],
                        "files": entry["files"],
                    }
                )

    # Write the filtered tasks to a new file
    with open("processed_data/translation/english_french_tasks.txt", "w") as outfile:
        for task in english_french_tasks:
            for file in task["files"]:
                outfile.write(f"{file}\n")


def get_english_tasks(
    language_pairs_file="processed_data/translation/language_pairs_output.jsonl", output_languages=["Spanish"]
):
    def is_english_x_task(entry, output_language):
        return (entry["input_language"] == "English" and entry["output_language"] == output_language) or (
            entry["input_language"] == output_language and entry["output_language"] == "English"
        )

    english_tasks = defaultdict(list)

    total_instances = 0

    for output_language in output_languages:
        task_instance_counts = {}
        language_instances = 0

        # Read the input file and filter tasks
        with open(language_pairs_file, "r") as infile:
            for line in infile:
                entry = json.loads(line)
                if is_english_x_task(entry, output_language):
                    english_tasks[output_language].extend(entry["files"])
                    logger.info(f"Found {len(entry['files'])} tasks for {output_language}")

        # Process each task file
        for task_file in english_tasks[output_language]:
            full_path = os.path.join("natural-instructions/tasks", task_file)
            try:
                with open(full_path, "r") as task_data_file:
                    task_data = json.load(task_data_file)

                    # if there are no positive or negative examples, we discard this task
                    if not task_data.get("Positive Examples") and not task_data.get("Negative Examples"):
                        logger.info(f"Discarding task {task_file} because it has no examples")
                        continue

                    instance_count = len(task_data["Instances"])
                    task_instance_counts[task_file] = instance_count
                    language_instances += instance_count
                    total_instances += instance_count
            except FileNotFoundError:
                logger.error(f"Warning: File not found: {full_path}")
            except json.JSONDecodeError:
                logger.error(f"Warning: Invalid JSON in file: {full_path}")
            except KeyError:
                logger.error(f"Warning: 'instances' key not found in file: {full_path}")

        logger.error(f"Total number of instances for {output_language}: {language_instances}")

    logger.error(f"Total number of instances across all tasks: {total_instances}")

    return english_tasks


def process_translation(output_languages=["Spanish", "French"], overwrite=False, verbose=False):
    process_task(save_prompt_lengths=False, save_prompts=False, overwrite=True, verbose=verbose)
    get_languages()
    task_dict = get_english_tasks(output_languages=output_languages)
    task_list = []
    for language in task_dict:
        task_list.extend(task_dict[language])
    process_task(save_prompt_lengths=False, save_prompts=True, task_file_list=task_list, overwrite=overwrite)


def evaluate_translations(
    model_name: Optional[str] = None,
    seed: Optional[str] = None,
    model_dir: Optional[str] = None,
    metric: str = "bleu",
    overwrite=True,
    use_wandb=False,
    entity="LLM_Accountability",
    gen_dir="processed_data/translation_model_outputs",
    output_dir="processed_data/translation_model_scores",
    verbose=True,
):
    data = None
    if (model_name is None or seed is None) and model_dir is None:
        raise ValueError("Either model_name and seed or dir must be provided.")

    if not model_name:
        split = model_dir.split("_seed")
        model_name = split[0]
        seed = f"seed{split[1]}"

    if use_wandb:
        wandb.init(
            project=f"{metric}_evaluation",
            entity=entity,
            name=create_run_string(),
            config={"model_name": model_name, "seed": seed},
            tags=["evaluate_model"],
        )

    gen_dir = f"{gen_dir}/{model_name}_{seed}"
    score_dir = f"{output_dir}/{model_name}_{seed}"

    # check if folder exists already
    if not Path(score_dir).exists():
        Path(score_dir).mkdir(parents=True, exist_ok=True)
    score_path = Path(score_dir) / f"{metric}_scores.json"
    cont_path = Path(gen_dir) / f"{model_name}_continuations_{seed}.json"

    # Load the evaluation metrics
    if metric == "bleu":
        score_function = load("bleu")
    else:
        score_function = load("rouge")

    if overwrite or not os.path.exists(score_path):
        with open(cont_path, "r", encoding="utf=8") as f:
            data = json.load(f)

        # Extract metadata and translations
        metadata = data["metadata"]
        machine_translations = data["continuations"]
        ground_truths = data["ground_truth"]

        scores = []

        # Evaluate each translation
        for mt, gt in zip(machine_translations, ground_truths):
            # Calculate BLEU score
            score = score_function.compute(predictions=[mt], references=[gt])

            if metric == "bleu":
                scores.append(score["bleu"])
            else:
                scores.append(score["rougeLsum"])

        # Prepare the output data
        output_data = {"metadata": metadata, f"{metric}_scores": scores}

        # Save the results to a new JSON file
        with open(score_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # data_path = "/root/accountability/data/tasks"  # TODO: get rid of root again
    # out_path = "/root/accountability/processed_data"
    # category_path = "/root/accountability/processed_data/categories"

    # process_translation()
    # task_dict = get_english_tasks(output_languages=["Spanish", "French"])
    # task_list = task_dict["Spanish"] + task_dict["French"]
    # process_task(save_prompt_lengths=False, save_prompts=True, task_file_list=task_list, overwrite=True)

    # process_translation(overwrite=True)
    # analyze_long_prompts("processed_data/translation/translation_data_few_shot.jsonl")
    # analyze_long_prompts("processed_data/translation/translation_data.jsonl")

    evaluate_translations(model_name="Meta-Llama-3-8B-Instruct", seed="seed2000", overwrite=True, use_wandb=False)
