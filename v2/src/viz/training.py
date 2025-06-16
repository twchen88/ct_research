import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Callable, Tuple

"""
src/viz/training.py
------------------------
This module provides functions for visualizing loss curves and saving plots.
"""

def save_plot(plot, path: str):
    """
    Save the plot to the specified path.
    
    Parameters:
        plot: The plot to save.
        path (str): The path where the plot will be saved.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plot.savefig(path)
    print(f"Plot saved to {path}")


def plot_single_curve(fig, train_loss, val_loss, output_path):
    """
    Plot a single curve for training and validation loss.
    
    Parameters:
        fig: The figure to plot on.
        train_loss (list): The training loss values.
        val_loss (list): The validation loss values.
    """
    plt.figure(fig.number)
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    save_plot(plt, output_path)

'''
# plot average improvement plots and store, d_type= Ground Truth or Prediction, mode=train or test, cur_score=whatever we need, data=test or train data
def plot_average_improvements(d_type, mode, cur_score, data):
    cur_score = np.nan_to_num(cur_score, nan=0)
    prev_score = np.nan_to_num(data[score_columns].copy().to_numpy(), nan=0)
    # Step 1: Compute differences
    differences = cur_score - prev_score
    # Step 2: Mask the differences using the encoding array
    masked_differences = np.where(data[encoding_columns].copy().to_numpy() == 1, differences, 0)  # Retain differences only where encoding is 1
    # Step 3: Compute the column-wise sum and count
    column_sums = np.sum(masked_differences, axis=0)  # Sum of differences for each column
    column_counts = np.sum(data[encoding_columns].copy().to_numpy(), axis=0)          # Number of 1s in each column
    # Step 4: Filter out columns with no encoding == 1
    valid_columns = column_counts > 0  # Boolean mask for valid columns
    filtered_sums = column_sums[valid_columns]
    filtered_counts = column_counts[valid_columns]
    # Step 5: Compute the column-wise averages for valid columns
    filtered_averages = filtered_sums / filtered_counts
    filtered_column_indices = np.where(valid_columns)[0]
    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))  # Create the figure and axes
    bars = ax.bar(range(len(filtered_averages)), filtered_averages, tick_label=[f"{i+1}" for i in filtered_column_indices])
    # Add values to the bars
    ax.bar_label(bars, fmt='%.4f', label_type='edge')
    # Set the y-axis range
    ax.set_ylim(-0.1, 0.1)
    # Add labels and title
    title_s = "%s %s Data Domain Improvement Averages" % (d_type, mode)
    plt.xlabel("Domains", fontsize=12)
    plt.ylabel("Average Difference", fontsize=12)
    plt.title(title_s, fontsize=16)
    plt.tight_layout()
    # Save the plot
    plt.savefig(output_dir + title_s + ".png")


def plot_mean_and_std(data, color_choice, setting=""):
    # Convert data to a NumPy array for easier manipulation
    data_array = np.array(data)
    
    # Calculate mean and standard deviation
    means = np.mean(data_array, axis=0)
    stds = np.std(data_array, axis=0)
    
    # Create the x-axis values
    x_values = np.arange(len(means))
    
    # Plotting
    plt.plot(x_values, means, label='%s Mean' % setting, color=color_choice)  # Mean line
    plt.fill_between(x_values, means - stds, means + stds, color=color_choice, alpha=0.2, label='%s Standard Deviation' % setting)
    
    plt.title('Mean and Standard Deviation Plot')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
'''