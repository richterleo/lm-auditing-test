import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

# Sample data with specified groups and models
data = pd.DataFrame(
    {
        "Model": [
            "Model_L",
            "Model_M1",
            "Model_M2",
            "Model_M3",
            "Model_M4",
            "Model_M5",
            "Model_R",
        ],
        "Group": [
            "Group1",
            "Group2",
            "Group2",
            "Group2",
            "Group2",
            "Group2",
            "Group3",
        ],
        "difference in mean toxicity": [
            0.0677124045253898 - 0.06774706010988371,
            0.09064904295864847 - 0.06774706010988371,
            0.08653019197064651 - 0.06774706010988371,
            0.11428576899347434 - 0.06774706010988371,
            0.08938699357564416 - 0.06774706010988371,
            0.09872509688922614 - 0.06774706010988371,
            0.10383416048410893 - 0.06774706010988371,
        ],
        "Wasserstein distance": [
            0.003294397455167,
            0.025835104952675852,
            0.01956497683683252,
            0.046398137443984126,
            0.0248757294017556,
            0.0321866051637882,
            0.03642399045212926,
        ],
        "Neural Net distance": [
            0.0035982596404209753,
            0.0503472856,
            0.027885657743900043,
            0.07638163288256687,
            0.047497452888637735,
            0.060533544264035254,
            0.06834232582734823,
        ],
    }
)

# Normalize the axes for alignment
normalized_data = data.copy()
metrics = ["difference in mean toxicity", "Wasserstein distance", "Neural Net distance"]

# Normalize each metric individually
for metric in metrics:
    scaler = MinMaxScaler()
    normalized_data[metric] = scaler.fit_transform(data[[metric]])

# Define the colors per model
model_colors = {
    "Model_L": "gray",
    "Model_R": "black",
    "Model_M1": "#1f77b4",
    "Model_M2": "#ff7f0e",
    "Model_M3": "#2ca02c",
    "Model_M4": "#d62728",
    "Model_M5": "#9467bd",
}

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 2))
ax.axis("off")  # Remove axes

# Define positions for the axes (from top to bottom)
num_metrics = len(metrics)
spacing = 0.05  # Adjusted spacing for close axes
axis_positions = [1 - i * spacing for i in range(num_metrics)]
metric_positions = dict(zip(metrics, axis_positions))

# Calculate tick label offset based on spacing
tick_label_offset = spacing * 0.3  # Adjusted for better visibility

# Define x-axis margins
x_margin = 0.02  # 2% margin on both sides
left_limit = 0 - x_margin
right_limit = 1 + x_margin

# Plot each metric axis
for metric in metrics:
    y = metric_positions[metric]
    # Draw horizontal line for the metric axis, extended
    ax.hlines(y, xmin=left_limit, xmax=right_limit, color="black", linewidth=0.8)
    # Label the metric axis
    ax.text(
        left_limit - 0.01,
        y,
        metric,
        horizontalalignment="right",
        verticalalignment="center",
        fontsize=12,
    )
    # Add x-axis ticks and labels
    metric_min = data[metric].min()
    metric_max = data[metric].max()
    tick_positions = np.linspace(0, 1, 5)
    tick_labels = [f"{metric_min + t * (metric_max - metric_min):.2f}" for t in tick_positions]
    for tx, tl in zip(tick_positions, tick_labels):
        ax.text(
            tx,
            y - tick_label_offset,
            tl,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=8,
        )

for idx, row in normalized_data.iterrows():
    model = row["Model"]
    color = model_colors[model]
    for metric in metrics:
        y = metric_positions[metric]
        x = row[metric]
        marker = "p" if model == "Model_R" else "X"  # Use pentagon for Model_R
        ax.plot(
            x,
            y,
            marker=marker,
            color=color,
            markersize=10,
            alpha=0.7,
        )


# Create custom legend entries
# For Group2, we'll create a multicolored 'X' marker using a custom handler
class HandlerMultiColoredX(HandlerBase):
    def __init__(self, colors, **kw):
        super().__init__(**kw)
        self.colors = colors

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        markers = []
        num_colors = len(self.colors)
        marker_size = fontsize * 1  # Adjust marker size as needed
        x_center = (width - xdescent) / 2
        y_center = (height - ydescent) / 2
        # Slight offsets to overlap the markers
        offsets = np.linspace(-marker_size / 1.5, marker_size / 1.5, num_colors)
        for i, (color, offset) in enumerate(zip(self.colors, offsets)):
            line = Line2D(
                [x_center + offset],
                [y_center],
                marker="X",
                markersize=marker_size,
                markerfacecolor=color,
                markeredgecolor=color,
                linestyle="None",
                alpha=0.9,
                transform=trans,
            )
            markers.append(line)
        return markers


# Colors for the models in Group2
group2_models = data[data["Group"] == "Group2"]["Model"]
group2_colors = [model_colors[model] for model in group2_models]

# Legend elements
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="X",
        color="gray",
        label="sampling variation",
        markerfacecolor="gray",
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        label="instruction tuning",
    ),
    Line2D(
        [0],
        [0],
        marker="p",
        color="black",
        label="Uncensored Llama3",
        markerfacecolor="black",
        markersize=10,
    ),
]

# Add legend to the right of the plot
ax.legend(
    handles=legend_elements,
    bbox_to_anchor=(1.05, 0.85),
    frameon=True,
    handler_map={legend_elements[1]: HandlerMultiColoredX(group2_colors)},
)

# Adjust plot limits
ax.set_xlim(left_limit, right_limit)
ax.set_ylim(min(axis_positions) - spacing * 1.5, max(axis_positions) + spacing * 0.5)

plt.tight_layout()
plt.savefig("means_and_distances.pdf", format="pdf", bbox_inches="tight")
# plt.show()
