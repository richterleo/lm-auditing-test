# import transformers
# import torch

# model_id = "aaditya/OpenBioLLM-Llama3-8B"

# tokenizer = transformers.AutoTokenizer.from_pretrained("meta-Ll")

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="auto",
# )

# messages = [
#     {
#         "role": "system",
#         "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience.",
#     },
#     {"role": "user", "content": "How can i split a 3mg or 4mg waefin pill so i can get a 2.5mg pill?"},
# ]

# prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# outputs = pipeline(
#     prompt,
#     max_new_tokens=256,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.0,
#     top_p=0.9,
# )
# print(outputs[0]["generated_text"][len(prompt) :])


import json
import matplotlib.pyplot as plt

# Read the JSON file
with open("dataset_lengths", "r") as file:
    data = json.load(file)

# Find the maximum value and its index
max_value = max(data)
max_index = data.index(max_value)

# Create a plot
plt.figure(figsize=(12, 6))
plt.plot(data, marker="o")
plt.title("Values Over Time")
plt.xlabel("Index")
plt.ylabel("Value")

# Highlight the maximum point
plt.plot(max_index, max_value, "ro", markersize=10, label=f"Max: {max_value} at index {max_index}")
plt.legend()

# Add grid lines
plt.grid(True, linestyle="--", alpha=0.7)

# Save the plot
plt.savefig("values_over_time.png")

# Print the results
print(f"Maximum value: {max_value}")
print(f"Index of maximum value: {max_index}")

print("Plot saved as 'values_over_time.png'")
