import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data provided
data = [
    {
        "model": "Meta-Llama-3-8B-Instruct_seed1000",
        "mean": 0.06774706010988371,
        "std": 0.08551269447414317,
        "median": 0.03357032,
    },
    {
        "model": "Llama-3-8B-ckpt1_seed2000",
        "mean": 0.0685489768938121,
        "std": 0.08684917401607557,
        "median": 0.033334624,
    },
    {
        "model": "Llama-3-8B-ckpt2_seed2000",
        "mean": 0.09739463288090477,
        "std": 0.14540491583233606,
        "median": 0.031213367,
    },
    {
        "model": "Llama-3-8B-ckpt3_seed2000",
        "mean": 0.10043781852884438,
        "std": 0.15548368086726927,
        "median": 0.02873857,
    },
    {
        "model": "Llama-3-8B-ckpt4_seed2000",
        "mean": 0.10162327301745495,
        "std": 0.15744501912466005,
        "median": 0.028856417,
    },
    {
        "model": "Llama-3-8B-ckpt5_seed1000",
        "mean": 0.10274904658214674,
        "std": 0.16187058311865288,
        "median": 0.028149333,
    },
    {
        "model": "Llama-3-8B-ckpt6_seed1000",
        "mean": 0.10247555553808989,
        "std": 0.1607670304682832,
        "median": 0.028385026,
    },
    {
        "model": "Llama-3-8B-ckpt7_seed1000",
        "mean": 0.10499873285420548,
        "std": 0.16488282699840698,
        "median": 0.02873857,
    },
    {
        "model": "Llama-3-8B-ckpt8_seed1000",
        "mean": 0.10804848934644057,
        "std": 0.16713585922884583,
        "median": 0.029799197,
    },
    {
        "model": "Llama-3-8B-ckpt9_seed1000",
        "mean": 0.11072725871393425,
        "std": 0.16915733369888958,
        "median": 0.030741978,
    },
    {
        "model": "Llama-3-8B-ckpt10_seed1000",
        "mean": 0.10897405406367779,
        "std": 0.1684587916523859,
        "median": 0.03015274,
    },
    {
        "model": "gemma-1.1-7b-it_seed1000",
        "mean": 0.04741929786317833,
        "std": 0.06296497290535912,
        "median": 0.024260364,
    },
    {
        "model": "gemma-1.1-7b-it-ckpt1_seed1000",
        "mean": 0.048893643680983485,
        "std": 0.06510317863557265,
        "median": 0.024731753,
    },
    {
        "model": "gemma-1.1-7b-it-ckpt2_seed1000",
        "mean": 0.0708336557642285,
        "std": 0.09591968682623654,
        "median": 0.030506283,
    },
    {
        "model": "gemma-1.1-7b-it-ckpt3_seed1000",
        "mean": 0.10628886279465986,
        "std": 0.13038735443861216,
        "median": 0.048099842,
    },
    {
        "model": "gemma-1.1-7b-it-ckpt4_seed1000",
        "mean": 0.11753490471059998,
        "std": 0.14871539191265382,
        "median": 0.04785245,
    },
    {
        "model": "gemma-1.1-7b-it-ckpt5_seed1000",
        "mean": 0.11171006926504434,
        "std": 0.1580448316815482,
        "median": 0.036634352,
    },
    {
        "model": "gemma-1.1-7b-it-ckpt6_seed1000",
        "mean": 0.11248987108999496,
        "std": 0.16135863834204675,
        "median": 0.036162965,
    },
    {
        "model": "gemma-1.1-7b-it-ckpt7_seed1000",
        "mean": 0.1149138650185123,
        "std": 0.16852556943797076,
        "median": 0.03545588,
    },
    {
        "model": "gemma-1.1-7b-it-ckpt8_seed1000",
        "mean": 0.12307730299399201,
        "std": 0.17862439279999953,
        "median": 0.03734144,
    },
    {
        "model": "gemma-1.1-7b-it-ckpt9_seed1000",
        "mean": 0.13691070564662403,
        "std": 0.1902800866547756,
        "median": 0.042657252,
    },
    {
        "model": "gemma-1.1-7b-it-ckpt10_seed1000",
        "mean": 0.141073854957449,
        "std": 0.19943546317250246,
        "median": 0.04290464,
    },
    {
        "model": "Mistral-7B-Instruct-v0.2_seed1000",
        "mean": 0.05708434783007121,
        "std": 0.0901178303528323,
        "median": 0.022021262,
    },
    {
        "model": "Mistral-7B-Instruct-ckpt1_seed1000",
        "mean": 0.0590593159080473,
        "std": 0.09356269752213643,
        "median": 0.022139108,
    },
    {
        "model": "Mistral-7B-Instruct-ckpt2_seed1000",
        "mean": 0.09741941657917084,
        "std": 0.1475779777666964,
        "median": 0.028856417,
    },
    {
        "model": "Mistral-7B-Instruct-ckpt3_seed1000",
        "mean": 0.11640168017377019,
        "std": 0.17137079191066842,
        "median": 0.03545588,
    },
    {
        "model": "Mistral-7B-Instruct-ckpt4_seed1000",
        "mean": 0.10425781682898609,
        "std": 0.16173379668120524,
        "median": 0.02920996,
    },
    {
        "model": "Mistral-7B-Instruct-ckpt5_seed1000",
        "mean": 0.10029005369077557,
        "std": 0.1565821477665735,
        "median": 0.028385026,
    },
    {
        "model": "Mistral-7B-Instruct-ckpt6_seed1000",
        "mean": 0.10650007630219185,
        "std": 0.16176062201575478,
        "median": 0.032863233,
    },
    {
        "model": "Mistral-7B-Instruct-ckpt7_seed1000",
        "mean": 0.10317066772091156,
        "std": 0.1612394505873458,
        "median": 0.028856417,
    },
    {
        "model": "Mistral-7B-Instruct-ckpt8_seed1000",
        "mean": 0.10621594309425778,
        "std": 0.16587916378484702,
        "median": 0.02873857,
    },
    {
        "model": "Mistral-7B-Instruct-ckpt9_seed1000",
        "mean": 0.10414101305339825,
        "std": 0.16654458542225345,
        "median": 0.027560094,
    },
    {
        "model": "Mistral-7B-Instruct-ckpt10_seed1000",
        "mean": 0.09912008898419702,
        "std": 0.1599541278204443,
        "median": 0.026499467,
    },
]


# Create a DataFrame from the data
df = pd.DataFrame(data)

# Define the models in the desired order along with their order numbers
# selected_models = [
#     ("Meta-Llama-3-8B-Instruct_seed1000", 0),
#     ("Llama-3-8B-ckpt1_seed2000", 1),
#     ("Llama-3-8B-ckpt2_seed2000", 2),
#     ("Llama-3-8B-ckpt3_seed2000", 3),
#     ("Llama-3-8B-ckpt4_seed2000", 4),
#     ("Llama-3-8B-ckpt5_seed1000", 5),
#     ("Llama-3-8B-ckpt6_seed1000", 6),
#     ("Llama-3-8B-ckpt7_seed1000", 7),
#     ("Llama-3-8B-ckpt8_seed1000", 8),
#     ("Llama-3-8B-ckpt9_seed1000", 9),
#     ("Llama-3-8B-ckpt10_seed1000", 10),
# ]

# selected_models = [
#     ("gemma-1.1-7b-it_seed1000", 0),
#     ("gemma-1.1-7b-it-ckpt1_seed1000", 1),
#     ("gemma-1.1-7b-it-ckpt2_seed1000", 2),
#     ("gemma-1.1-7b-it-ckpt3_seed1000", 3),
#     ("gemma-1.1-7b-it-ckpt4_seed1000", 4),
#     ("gemma-1.1-7b-it-ckpt5_seed1000", 5),
#     ("gemma-1.1-7b-it-ckpt6_seed1000", 6),
#     ("gemma-1.1-7b-it-ckpt7_seed1000", 7),
#     ("gemma-1.1-7b-it-ckpt8_seed1000", 8),
#     ("gemma-1.1-7b-it-ckpt9_seed1000", 9),
#     ("gemma-1.1-7b-it-ckpt10_seed1000", 10),
# ]

selected_models = [
    ("Mistral-7B-Instruct-v0.2_seed1000", 0),
    ("Mistral-7B-Instruct-ckpt1_seed1000", 1),
    ("Mistral-7B-Instruct-ckpt2_seed1000", 2),
    ("Mistral-7B-Instruct-ckpt3_seed1000", 3),
    ("Mistral-7B-Instruct-ckpt4_seed1000", 4),
    ("Mistral-7B-Instruct-ckpt5_seed1000", 5),
    ("Mistral-7B-Instruct-ckpt6_seed1000", 6),
    ("Mistral-7B-Instruct-ckpt7_seed1000", 7),
    ("Mistral-7B-Instruct-ckpt8_seed1000", 8),
    ("Mistral-7B-Instruct-ckpt9_seed1000", 9),
    ("Mistral-7B-Instruct-ckpt10_seed1000", 10),
]


# Create a DataFrame with the selected models and their orders
selected_df = pd.DataFrame(selected_models, columns=["model", "Order"])

# Merge this with the original df to get the data
df_selected = pd.merge(selected_df, df, on="model")


# Create labels for the y-axis
def label_row(row):
    if row["Order"] == 0:
        return "baseline"
    else:
        return f"ckpt {row['Order']}"


df_selected["Label"] = df_selected.apply(label_row, axis=1)

# Now, sort the DataFrame by 'Order'
df_selected.sort_values(by="Order", inplace=True)

# Generate colors: baseline in white and checkpoints getting darker
n_bars = len(df_selected)
# Generate a reversed 'viridis' palette with the baseline color set to white
palette = sns.color_palette("viridis", n_colors=n_bars - 1)
palette = palette[::-1]
palette = [(1, 1, 1)] + palette  # Set the first color (baseline) to white

# Set the style to match the example plot
sns.set_style("white")
plt.figure(figsize=(12, 6))

# Create the horizontal bar plot
ax = sns.barplot(
    x="mean",
    y="Label",
    data=df_selected,
    palette=palette,
    edgecolor="black",
    linewidth=1.5,
)

# Ensure all spines are visible and set their linewidths
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2.5)

# Customize the plot
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("mean toxicity", fontsize=18)
plt.ylabel("", fontsize=18)

# Add horizontal grid lines
# ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#ddddee", zorder=0)

# x_min, x_max = ax.get_xlim()
# x_ticks = np.arange(np.ceil(x_min * 100) / 100, x_max + 0.01, 0.0)  # Added 0.01 to x_max to include the last tick
# ax.set_xticks(x_ticks, minor=True)
# ax.xaxis.grid(True, which="minor", linestyle="--", linewidth=0.5, color="#ddddee", zorder=0)

# # Adjust x-axis limits to ensure all vertical lines are visible
# ax.set_xlim(x_min, np.ceil(x_max * 100) / 100)

# Remove y-axis grid
ax.yaxis.grid(False)

# Adjust layout for tightness
plt.tight_layout()

# Save the plot
plt.savefig("mean_toxicity_for_checkpoints_mistral.pdf", format="pdf", bbox_inches="tight")
# plt.show()
