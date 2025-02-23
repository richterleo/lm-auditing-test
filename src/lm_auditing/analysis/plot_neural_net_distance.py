import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os


def read_csv_files(model_names):
    all_data = []
    base_path = "/root/Auditing_test_for_LMs/Auditing_test_for_LMs/test_outputs/Meta-Llama-3-8B-Instruct_seed1000_{}"
    file_pattern = "distance_scores_100_1000_300*.csv"

    for model_name in model_names:
        model_path = base_path.format(model_name)
        csv_files = glob.glob(os.path.join(model_path, file_pattern))

        if not csv_files:
            print(f"No matching CSV file found for model: {model_name}")
            continue

        # Use the first matching file
        file_path = csv_files[0]
        df = pd.read_csv(file_path)
        df["Model"] = model_name
        all_data.append(df)

    if not all_data:
        raise ValueError("No data found for any of the provided models.")

    return pd.concat(all_data, ignore_index=True)


def calculate_standard_deviations(data):
    # Function to group the largest sample sizes
    def group_max_samples(x):
        if x >= 97000:  # Threshold for grouping largest samples
            return "max"
        return x

    # Apply the grouping function
    data["grouped_samples"] = data["num_train_samples"].apply(group_max_samples)

    std_devs = data.groupby(["Model", "grouped_samples"])["NeuralNet"].std().unstack(level="grouped_samples")

    print("Standard Deviations:")
    for model in std_devs.index:
        print(f"\n{model}:")
        for samples, std in std_devs.loc[model].items():
            if samples == "max":
                print(f"  Maximum samples (~98k): {std:.6f}")
            else:
                print(f"  {samples} samples: {std:.6f}")

        # Calculate percentage decrease
        first_std = std_devs.loc[model].iloc[0]
        last_std = std_devs.loc[model].iloc[-1]
        if pd.notna(first_std) and pd.notna(last_std) and first_std != 0:
            percent_decrease = (first_std - last_std) / first_std * 100
            print(f"  Percentage decrease: {percent_decrease:.2f}%")
        else:
            print("  Percentage decrease: Cannot be calculated (NaN or zero values)")


def create_plot(data, legend_names):
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid", {"grid.color": ".9"})  # Lighter grid color

    # Create the plot
    ax = sns.lineplot(
        x="num_train_samples", y="NeuralNet", hue="Model", data=data, marker="X", markersize=10, palette="husl"
    )

    # Set only x-axis to log scale
    plt.xscale("log")
    plt.yscale("linear")  # Ensure y-axis is linear

    # Remove title
    plt.title("")

    # Update axis labels
    plt.xlabel("number of training samples (log scale)", fontsize=18)
    plt.ylabel("estimated neural net distance", fontsize=18)

    # Increase tick label size
    plt.tick_params(axis="both", which="major", labelsize=14)

    # Add a thick black box around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    updated_labels = [legend_names.get(label, label) for label in labels]
    plt.legend(handles, updated_labels, loc="best", fontsize="14", frameon=True, framealpha=0.8)

    plt.tight_layout()
    plt.savefig("neural_net_distance_plot.pdf", format="pdf", bbox_inches="tight")


# List of model names
model_names = [
    "Llama-3-8B-ckpt1_seed2000",
    "Llama-3-8B-ckpt5_seed1000",
    "Llama-3-8B-ckpt10_seed1000",
    "Meta-Llama-3-8B-Instruct-hightemp_seed1000",
    "Meta-Llama-3-8B-Instruct_seed2000",
]

# Dictionary for custom legend names
legend_names = {
    "Llama-3-8B-ckpt1_seed2000": "checkpoint 1",
    "Llama-3-8B-ckpt5_seed1000": "checkpoint 5",
    "Llama-3-8B-ckpt10_seed1000": "checkpoint 10",
    "Meta-Llama-3-8B-Instruct-hightemp_seed1000": "Llama3 with varied sampling",
    "Meta-Llama-3-8B-Instruct_seed2000": "Llama3 with different seed",
}

try:
    # Read CSV files for the specified models
    data = read_csv_files(model_names)

    calculate_standard_deviations(data)

    # Create the plot
    create_plot(data, legend_names)
except Exception as e:
    print(f"An error occurred: {str(e)}")
