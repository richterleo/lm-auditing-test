import os
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

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
):
    data_path = Path(base_dir).parent / Path(data_path)
    out_path = Path(base_dir).parent / Path(out_path)
    category_path = out_path / task
    os.makedirs(category_path, exist_ok=True)

    prompt_length_dict = {"bare_data": [], "few_shot_data": []}

    bare_prompts = os.path.join(category_path, f"{task}_data.jsonl")
    few_shot_prompts = os.path.join(category_path, f"{task}_data_few_shot.jsonl")

    if os.path.exists(bare_prompts) and os.path.exists(few_shot_prompts) and not overwrite:
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
                        current_categories = data.get("Categories", [])
                        current_categories = [cat.lower().replace(" ", "_") for cat in current_categories]
                        if task in current_categories:
                            instruction = data.get("Definition", [None])[0]
                            instances = data.get("Instances", [])
                            for instance in instances:
                                instance_input = instance.get("input", "")
                                instance_outputs = instance.get("output", [])
                                prompt = create_prompt(data, instance_input)
                                output = random.choice(instance_outputs)
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
                logger.info(f"Average prompt length for {key}: {avg_length}")

        logger.info(f"Creating files for {task} data...")

        with open(bare_prompts, "w") as f:
            for entry in bare_data:
                f.write(json.dumps(entry) + "\n")

        with open(few_shot_prompts, "w") as f:
            for entry in few_shot_data:
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Files saved to {bare_prompts} and {few_shot_prompts}")


if __name__ == "__main__":
    # data_path = "/root/accountability/data/tasks"  # TODO: get rid of root again
    # out_path = "/root/accountability/processed_data"
    # category_path = "/root/accountability/processed_data/categories"

    process_task()
