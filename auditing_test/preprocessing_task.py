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
    task_file_list=None,
    save_prompts=True,
):
    base_dir = Path.cwd()
    data_path = Path(base_dir) / Path(data_path)
    out_path = Path(base_dir) / Path(out_path)
    category_path = out_path / task
    os.makedirs(category_path, exist_ok=True)
    prompt_length_dict = {"bare_data": [], "few_shot_data": []}

    if task == "translation":
        language_dict = {}
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

            logger.info(f"Files saved to {bare_prompts} and {few_shot_prompts}")


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
                    print(f"Found task for {output_language}")
                    english_tasks[output_language].extend(entry["files"])
                    print(f"Found {len(entry['files'])} tasks for {output_language}")

        # Process each task file
        for task_file in english_tasks[output_language]:
            full_path = os.path.join("natural-instructions/tasks", task_file)
            try:
                with open(full_path, "r") as task_data_file:
                    task_data = json.load(task_data_file)
                    instance_count = len(task_data["Instances"])
                    task_instance_counts[task_file] = instance_count
                    language_instances += instance_count
                    total_instances += instance_count
            except FileNotFoundError:
                print(f"Warning: File not found: {full_path}")
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in file: {full_path}")
            except KeyError:
                print(f"Warning: 'instances' key not found in file: {full_path}")

        print(f"Total number of instances for {output_language}: {language_instances}")

    print(f"Total number of instances across all tasks: {total_instances}")

    return english_tasks


if __name__ == "__main__":
    # data_path = "/root/accountability/data/tasks"  # TODO: get rid of root again
    # out_path = "/root/accountability/processed_data"
    # category_path = "/root/accountability/processed_data/categories"

    process_task_new(save_prompt_lengths=False)
    get_languages()
    task_list = get_english_tasks()
    print(task_list)
    process_task_new(save_prompt_lengths=False, task_file_list=task_list["Spanish"])
