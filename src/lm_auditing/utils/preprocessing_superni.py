import json
import logging
import random
import sys

from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional


class SNITranslationProcessor:
    """
    A processor for handling Super Natural Instructions (SNI) translation data.

    This class provides methods to preprocess translation tasks, create language dictionaries,
    filter tasks based on specified output languages, and manage the overall processing workflow.
    """

    PROMPT_DICT = {
        "prompt_input": (
            "### Instruction:\n"
            "{instruction}\n\n"
            "### Positive Examples:\n"
            "{positive_examples}\n\n"
            "### Negative Examples:\n"
            "{negative_examples}\n\n"
            "### Input:\n"
            "{input}\n\n"
            "### Output:\n"
        ),
        "system_prompt": (
            "You are a professional translator.\n"
            "Your job is to translate **exclusively** from {source_lang} into {target_lang}.\n"
            "Provide **only** the {target_lang} translation.\n"
            "Do not include any explanations, disclaimers, or additional commentary.\n\n"
            "### Input:\n"
            "{input}\n\n"
            "### Output:\n"
        ),
    }

    def __init__(
        self,
        data_path: str = "natural-instructions/tasks",
        output_path: str = "data",
        script_dir: Optional[str] = None,
        overwrite: bool = False,
        verbose: bool = True,
        category: str = "translation",
    ):
        """
        Initialize the SNITranslationProcessor.

        Args:
            data_path (str): Relative path to the directory containing task JSON files.
            output_path (str): Relative path to the directory where processed data will be saved.
            script_dir (Optional[str]): Path to the script directory. If None, defaults to the current file's directory.
            overwrite (bool): Whether to overwrite existing processed files.
            verbose (bool): Enable verbose logging.
        """
        self.script_dir = Path(__file__).resolve().parents[3] if script_dir is None else Path(script_dir)
        self.data_path = self.script_dir / data_path
        self.output_path = self.script_dir / output_path / category
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.language_dict_path = self.output_path / "language_dict.json"
        self.language_pairs_path = self.output_path / "language_pairs_output.jsonl"

        self.overwrite = overwrite
        self.verbose = verbose
        self.category = category

        # Set up logging
        self.logger = logging.getLogger(name=__file__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler("preprocessing_SNI.log")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def format_example(self, example: Dict[str, str], index: int) -> str:
        """
        Format a single example with its index.

        Args:
            example (Dict[str, str]): A dictionary containing 'input' and 'output' keys.
            index (int): The index of the example.

        Returns:
            str: A formatted string representing the example.
        """
        return f"{index}. Input: {example.get('input', '')}\n" f"   Output: {example.get('output', '')}\n"

    def format_content(
        self,
        definition: str,
        positive_examples: List[Dict[str, str]],
        negative_examples: List[Dict[str, str]],
        input_text: str,
    ) -> str:
        """
        Format the content for a prompt using definition and examples.

        Args:
            definition (str): The instruction or definition of the task.
            positive_examples (List[Dict[str, str]]): A list of positive example dictionaries.
            negative_examples (List[Dict[str, str]]): A list of negative example dictionaries.
            input_text (str): The input text for which the prompt is being created.

        Returns:
            str: A formatted prompt string.
        """
        formatted_positive = "\n".join(
            [self.format_example(example, i) for i, example in enumerate(positive_examples, start=1)]
        )
        formatted_negative = "\n".join(
            [self.format_example(example, i) for i, example in enumerate(negative_examples, start=1)]
        )
        return self.PROMPT_DICT["prompt_input"].format(
            instruction=definition,
            positive_examples=formatted_positive,
            negative_examples=formatted_negative,
            input=input_text,
        )

    def create_analog_prompt(self, task_data: Dict, instance_input: str) -> Optional[str]:
        """
        Create an analog prompt from task data and instance input.

        Args:
            task_data (Dict): The task data containing definitions and examples.
            instance_input (str): The input text for the instance.

        Returns:
            Optional[str]: The formatted prompt or None if no examples are available.
        """
        definition = task_data.get("Definition", [""])[0]
        positive_examples = task_data.get("Positive Examples", [])
        negative_examples = task_data.get("Negative Examples", [])

        if not (positive_examples or negative_examples):
            return None

        prompt = self.format_content(definition, positive_examples, negative_examples, instance_input).strip()
        return prompt

    def process_task(
        self,
        save_prompt_lengths: bool = True,
        task_file_list: Optional[List[str]] = None,
        save_prompts: bool = True,
        thresh_instance: int = 1800,
        thresh_prompt: int = 2200,
        overwrite_prompts=False,
    ):
        """
        Process translation tasks by reading JSON files, creating prompts, and saving processed data.

        Args:
            save_prompt_lengths (bool): Whether to save prompt length statistics.
            task_file_list (Optional[List[str]]): A list of specific task files to process. If None, process all.
            save_prompts (bool): Whether to save the processed prompts to files.
            thresh_instance (int): Maximum allowed length for instance input.
            thresh_prompt (int): Maximum allowed length for the generated prompt.
        """

        self.logger.info(f"Task file list: {task_file_list}")

        language_dict = {}
        bare_prompts_path = self.output_path / "translation_data.jsonl"
        few_shot_prompts_path = self.output_path / "translation_data_fewshot.jsonl"
        explicit_prompts_path = self.output_path / "translation_data_explicit.jsonl"

        if bare_prompts_path.exists() and few_shot_prompts_path.exists() and explicit_prompts_path.exists() and not overwrite_prompts:
            self.logger.info(f"Translation data already exists at {self.output_path}")
            return

        if not save_prompts:
            if self.language_dict_path.exists() and not self.overwrite:
                self.logger.info(f"Language dictionary already exists at {self.language_dict_path}")
                return

        prompt_length_dict = {"bare_data": [], "few_shot_data": [], "explicit_data": []}
        skipped_long_prompts = 0

        bare_data = []
        few_shot_data = []
        explicit_data = []

        for file_path in self.data_path.iterdir():
            if not file_path.suffix == ".json":
                continue

            with open(file_path, "r", encoding="utf-8") as json_file:
                try:
                    data = json.load(json_file)

                    # either we already have a list of subtasks to process ...
                    if task_file_list and file_path.name not in task_file_list:
                        continue

                    # ... or we process all subtasks of a specific category
                    else:
                        current_categories = data.get("Categories", [])
                        current_categories = [cat.lower().replace(" ", "_") for cat in current_categories]
                        if self.category not in current_categories:
                            continue

                    self.logger.info(f"Processing file: {file_path.name}")

                    instruction = data.get("Definition", [""])[0]
                    instances = data.get("Instances", [])
                    input_lang = data.get("Input_language", [""])[0]
                    output_lang = data.get("Output_language", [""])[0]

                    language_dict[file_path.name] = {
                        "input_language": input_lang,
                        "output_language": output_lang,
                    }

                    for instance in instances:
                        instance_input = instance.get("input", "")
                        instance_outputs = instance.get("output", [])
                        prompt = self.create_analog_prompt(data, instance_input)
                        if not prompt:
                            continue
                        output = random.choice(instance_outputs)

                        if len(instance_input) > thresh_instance or len(prompt) > thresh_prompt:
                            if self.verbose:
                                self.logger.info("Long prompt skipped")
                            skipped_long_prompts += 1
                            continue

                        bare_data.append({"instruction": instruction, "input": instance_input, "output": output})
                        few_shot_data.append({"prompt": prompt, "output": output})
                        
                        # Create explicit system prompt data
                        if input_lang and output_lang:
                            explicit_prompt = self.PROMPT_DICT["system_prompt"].format(
                                source_lang=input_lang,
                                target_lang=output_lang,
                                input=instance_input,
                            )
                            explicit_data.append({"prompt": explicit_prompt, "output": output})

                        prompt_length_dict["bare_data"].append(len(instance_input))
                        prompt_length_dict["few_shot_data"].append(len(prompt))

                except json.JSONDecodeError:
                    self.logger.error(f"Error decoding JSON from file: {file_path.name}")
                except Exception as e:
                    self.logger.error(f"Unexpected error processing file {file_path.name}: {e}")

        if save_prompt_lengths:
            for key in prompt_length_dict:
                lengths = prompt_length_dict[key]
                avg_length = sum(lengths) / len(lengths) if lengths else 0
                self.logger.info(f"Average prompt length for {key}: {avg_length:.2f}")

        self.logger.info("Creating files for translation data...")

        with open(self.language_dict_path, "w", encoding="utf-8") as f:
            json.dump(language_dict, f, ensure_ascii=False, indent=2)

        if save_prompts:
            with open(bare_prompts_path, "w", encoding="utf-8") as f:
                for entry in bare_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            with open(few_shot_prompts_path, "w", encoding="utf-8") as f:
                for entry in few_shot_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            with open(explicit_prompts_path, "w", encoding="utf-8") as f:
                for entry in explicit_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            self.logger.info(
                f"Files saved to {bare_prompts_path}, {few_shot_prompts_path}, and {explicit_prompts_path}"
            )

    def create_language_dict(self):
        """
        Create a language dictionary mapping language pairs to their corresponding task files.

        The language dictionary is saved as a JSONL file in the output directory.
        """

        if not self.language_dict_path.exists():
            self.logger.error(f"Language file does not exist: {self.language_dict_path}")
            sys.exit(1)

        with open(self.language_dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        language_pairs = defaultdict(list)

        for filename, info in data.items():
            input_lang = info.get("input_language", "")[0]
            output_lang = info.get("output_language", "")[0]
            pair = (input_lang, output_lang)
            language_pairs[pair].append(filename)

        if not self.language_pairs_path.exists() or self.overwrite:
            with open(self.language_pairs_path, "w", encoding="utf-8") as outfile:
                for pair, files in language_pairs.items():
                    json_object = {"input_language": pair[0], "output_language": pair[1], "files": files}
                    json.dump(json_object, outfile, ensure_ascii=False)
                    outfile.write("\n")

    def get_english_tasks(self, output_languages: List[str] = ["Spanish", "French"]) -> Dict[str, List[str]]:
        """
        Retrieve tasks that involve English as either input or output language for specified target languages.

        Args:
            output_languages (List[str]): List of target output languages to filter tasks.

        Returns:
            Dict[str, List[str]]: A dictionary mapping each output language to its corresponding task files.
        """
        english_tasks = defaultdict(list)
        total_instances = 0

        if not self.language_pairs_path.exists():
            self.logger.error(f"Language pairs file does not exist: {self.language_pairs_path}")
            return english_tasks

        for output_language in output_languages:
            language_instances = 0

            with open(self.language_pairs_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    entry = json.loads(line)
                    input_lang = entry.get("input_language", "")
                    output_lang = entry.get("output_language", "")
                    if (input_lang == "English" and output_lang == output_language) or (
                        input_lang == output_language and output_lang == "English"
                    ):
                        english_tasks[output_language].extend(entry.get("files", []))
                        self.logger.info(f"Found {len(entry.get('files', []))} tasks for {output_language}")

            for task_file in english_tasks[output_language]:
                full_path = self.data_path / task_file
                try:
                    with open(full_path, "r", encoding="utf-8") as task_data_file:
                        task_data = json.load(task_data_file)

                        if not task_data.get("Positive Examples") and not task_data.get("Negative Examples"):
                            self.logger.info(f"Discarding task {task_file} because it has no examples")
                            continue

                        instance_count = len(task_data.get("Instances", []))
                        language_instances += instance_count
                        total_instances += instance_count
                except FileNotFoundError:
                    self.logger.error(f"Warning: File not found: {full_path}")
                except json.JSONDecodeError:
                    self.logger.error(f"Warning: Invalid JSON in file: {full_path}")
                except KeyError:
                    self.logger.error(f"Warning: 'Instances' key not found in file: {full_path}")
                except Exception as e:
                    self.logger.error(f"Unexpected error processing file {full_path}: {e}")

            self.logger.info(f"Total number of instances for {output_language}: {language_instances}")

        self.logger.info(f"Total number of instances across all tasks: {total_instances}")

        return english_tasks

    def process_translation(self, output_languages: List[str] = ["Spanish", "French"]):
        """
        Execute the full translation processing workflow:
        1. Process all tasks to generate bare and few-shot prompts without saving them.
        2. Create a language dictionary based on the processed data.
        3. Retrieve tasks that involve English and specified output languages.
        4. Process only the filtered tasks to generate and save prompts.

        Args:
            output_languages (List[str]): List of target output languages for filtering tasks.
        """
        # Step 1: Process all tasks to get the language dictionary without saving prompts
        self.process_task(save_prompt_lengths=False, save_prompts=False, overwrite_prompts=True)

        # Step 2: Extract language pairs and create language dictionary
        self.create_language_dict()

        # Step 3: Get tasks for specified output languages
        task_dict = self.get_english_tasks(output_languages=output_languages)

        # Step 4: Build the task list
        task_list = [file_name for files in task_dict.values() for file_name in files]

        # Step 5: Process only the tasks in the task list and save prompts
        self.process_task(
            save_prompt_lengths=False,
            save_prompts=True,
            task_file_list=task_list,
            overwrite_prompts=self.overwrite,
        )


def analyze_long_prompts(file_path: str):
    """
    Analyze prompt lengths in a JSONL file and generate statistics and a plot.

    Args:
        file_path (str): Path to the JSONL file containing prompts.

    Generates:
        - A plot image showing prompt lengths over the dataset.
        - Prints average and maximum prompt lengths.
    """
    import matplotlib.pyplot as plt

    file_path = Path(file_path)

    def read_jsonl(fp: str) -> List[Dict[str, str]]:
        """Read a JSONL file and return a list of dictionaries."""
        with open(fp, "r", encoding="utf-8") as file:
            return [json.loads(line) for line in file]

    def get_prompt_length(item: Dict[str, str]) -> int:
        """Calculate the length of a prompt based on its content."""
        if "prompt" in item:
            return len(item["prompt"])
        elif "instruction" in item and "input" in item:
            return len(item["instruction"] + item["input"])
        elif "instruction" in item:
            return len(item["instruction"])
        else:
            raise ValueError("Unexpected data format")

    # Read the JSONL file
    try:
        data = read_jsonl(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        return

    # Calculate prompt lengths
    try:
        prompt_lengths = [get_prompt_length(item) for item in data]
    except ValueError as e:
        print(f"Error calculating prompt lengths: {e}")
        return

    if not prompt_lengths:
        print("No prompt lengths to analyze.")
        return

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
    plot_path = file_path.parent / f"prompt_lengths_{file_path.name}.png"
    plt.savefig(plot_path)
    plt.close()

    # Print results
    print(f"Average prompt length: {avg_length:.2f} characters")
    print(f"Maximum prompt length: {max_length} characters at index {max_index}")
    print(f"Plot saved as {plot_path}")


if __name__ == "__main__":
    # Initialize the processor with desired configurations
    processor = SNITranslationProcessor(overwrite=False, verbose=False)

    # Execute the translation processing workflow
    processor.process_translation(output_languages=["Spanish", "French"])

    # To analyze long prompts, uncomment the following lines and provide the correct file paths:
    # analyze_long_prompts("translation_data/translation_data_few_shot.jsonl")
    # analyze_long_prompts("translation_data/translation_data.jsonl")
