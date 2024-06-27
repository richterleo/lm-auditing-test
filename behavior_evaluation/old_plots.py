import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm

def step_function(x, sorted_samples):
    y = 0
    for i, sample in enumerate(sorted_samples):
        if x > sample:
            y = (i + 1) / len(sorted_samples)
    return y


def plot_one_sample_kolmogorov(n_samples=100, file_name=None, pknown="Gaussian", seed=42, display=False):
    
    np.random.seed(seed)

    # sample from pknown
    if pknown == "Gaussian":
        samples = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    
    
    sorted_samples = np.sort(samples)

    x_values = np.linspace(-3, 3, 400)

    y_Gaussian = norm.cdf(x_values, 0, 1)
    y_step = np.array([step_function(x, sorted_samples) for x in x_values])
    
    # Get largest diff value for drawing 
    diff = np.array([step_function(x, sorted_samples) for x in sorted_samples]) - norm.cdf(sorted_samples, 0, 1)
    index_of_max_diff = np.argmax(np.abs(diff))
    max_diff_value = diff[index_of_max_diff]
    max_diff_sample = sorted_samples[index_of_max_diff]
    
    # Plot the line showing the largest difference
    y_step_max_diff = step_function(max_diff_sample, sorted_samples)
    y_cdf_max_diff = norm.cdf(max_diff_sample, 0, 1)


    # Plotting both functions
    plt.figure(figsize=(10, 6))

    # Plot the step function
    plt.step(x_values, y_step, where='post', label='Empirical Cumulative Distribution Function', color='purple')

    # Plot the smooth function
    plt.plot(x_values, y_Gaussian, label='Known Cumulative Distribution Function', color='green')
    
    plt.plot([max_diff_sample, max_diff_sample], [y_step_max_diff, y_cdf_max_diff], color='red', linewidth=2, label='Max Difference')


    # Adding labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Hide x-axis labels
    plt.xticks([])
    
    plt.title('Empirical CDF and known CDF')
    plt.legend()

    # Display the plot
    if file_name:
        plt.savefig(file_name, bbox_inches='tight', dpi=300)
    
    if display:    
        plt.show()
    
    return (diff, index_of_max_diff, max_diff_value)

def plot_two_sample_kolmogorov(n_samples=100, m_samples=100, file_name=None, pknown="Gaussian", seed=42, display=False):
    
    np.random.seed(seed)

    # sample from pknown
    if pknown == "Gaussian":
        p_samples = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
        q_samples = np.random.normal(loc=0.0, scale=1.0, size=m_samples)
    
    
    sorted_p_samples = np.sort(p_samples)
    sorted_q_samples = np.sort(q_samples)

    # Define the step function


    x_values = np.linspace(-3, 3, 400)

    p_step = np.array([step_function(x, sorted_p_samples) for x in x_values])
    q_step = np.array([step_function(x, sorted_q_samples) for x in x_values])
    
    # Get largest diff value for drawing 
    # diff = np.array([step_function(x, sorted_samples) for x in sorted_samples]) - norm.cdf(sorted_samples, 0, 1)
    # index_of_max_diff = np.argmax(np.abs(diff))
    # max_diff_value = diff[index_of_max_diff]
    # max_diff_sample = sorted_samples[index_of_max_diff]
    
    # Plot the line showing the largest difference
    # y_step_max_diff = step_function(max_diff_sample, sorted_samples)
    # y_cdf_max_diff = norm.cdf(max_diff_sample, 0, 1)
    
    low_diff_p = np.array([step_function(sorted_p_samples[i], sorted_p_samples) for i in range(len(sorted_p_samples))]) - np.array([step_function(sorted_p_samples[i], sorted_q_samples) for i in range(len(sorted_p_samples))])
    high_diff_p = low_diff = np.array([step_function(sorted_p_samples[i], sorted_p_samples) for i in range(len(sorted_p_samples))]) - np.array([step_function(sorted_p_samples[i], sorted_q_samples) for i in range(len(sorted_p_samples))])


    # Plotting both functions
    plt.figure(figsize=(10, 6))

    # Plot the step function
    plt.step(x_values, p_step, where='post', label='Empirical Cumulative Distribution Function', color='purple')

    # Plot the smooth function
    plt.plot(x_values, q_step, label='Known Cumulative Distribution Function', color='green')
    
    # plt.plot([max_diff_sample, max_diff_sample], [y_step_max_diff, y_cdf_max_diff], color='red', linewidth=2, label='Max Difference')


    # Adding labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Hide x-axis labels
    plt.xticks([])
    
    plt.title('Empirical CDF and known CDF')
    plt.legend()

    # Display the plot
    if file_name:
        plt.savefig(file_name, bbox_inches='tight', dpi=300)
    
    if display:    
        plt.show()
    
    # return (diff, index_of_max_diff, max_diff_value)
    
if __name__ == "__main__":
    
    # diff, index, max_value = plot_one_sample_kolmogorov(file_name="Plots/Gaussian_Kolmogorov.png", n_samples=10)
    # print(f"These are the diffs: {diff}")
    # print(f"This is the index max: {index}")
    # print(f"This is the largest deviation: {max_value}")
    
    plot_two_sample_kolmogorov(n_samples=10, m_samples=5, file_name="Plots/Gaussian_Kolmogorov_two_sample.png")
    
    
