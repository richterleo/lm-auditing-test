import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Set the style and figure size
sns.set_style("whitegrid")
plt.figure(figsize=(12, 4))

# Parameters
mu = 1
sigma = 1
lambda_exp = 1
threshold = 0

# Generate data
x_normal = np.linspace(-2, 4, 1000)
x_exp = np.linspace(0, 6, 1000)
y_normal = stats.norm.pdf(x_normal, mu, sigma)
y_exp = stats.expon.pdf(x_exp, scale=1 / lambda_exp)

# Calculate probabilities below threshold
prob_normal = stats.norm.cdf(threshold, mu, sigma)
prob_exp = stats.expon.cdf(threshold, scale=1 / lambda_exp)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.3})


# Plot Normal Distribution
sns.lineplot(x=x_normal, y=y_normal, ax=ax1, color="skyblue")
ax1.fill_between(x_normal, y_normal, color="skyblue", alpha=0.5)
ax1.set_title("Normal Distribution")
ax1.set_xlabel("x")
ax1.set_ylabel("Probability Density")
ax1.set_ylim(0, 1)

# Add threshold line and probability text for Normal Distribution
ax1.axvline(threshold, color="red", linestyle="--", label="Threshold (x=0)")
ax1.fill_between(x_normal[x_normal <= threshold], y_normal[x_normal <= threshold], color="red", alpha=0.3)
ax1.text(
    0.05,
    0.95,
    f"P(X ≤ {threshold}) = {prob_normal:.4f}",
    transform=ax1.transAxes,
    verticalalignment="top",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# Plot Exponential Distribution
sns.lineplot(x=x_exp, y=y_exp, ax=ax2, color="skyblue")
ax2.fill_between(x_exp, y_exp, color="skyblue", alpha=0.5)
ax2.set_title("Exponential Distribution")
ax2.set_xlabel("x")
ax2.set_ylabel("Probability Density")
ax2.set_ylim(0, 1)

# Add threshold line and probability text for Exponential Distribution
ax2.axvline(threshold, color="red", linestyle="--", label="Threshold (x=0)")
ax2.fill_between(x_exp[x_exp <= threshold], y_exp[x_exp <= threshold], color="red", alpha=0.3)
ax2.text(
    0.05,
    0.95,
    f"P(X ≤ {threshold}) = {prob_exp:.4f}",
    transform=ax2.transAxes,
    verticalalignment="top",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# Add legends
ax1.legend()
ax2.legend()

# Adjust layout and display
# plt.tight_layout()
# plt.show()
plt.savefig("extra_plots/same_mean_std_different.pdf", format="pdf", bbox_inches="tight")
