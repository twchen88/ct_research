import matplotlib.pyplot as plt

"""
src/viz/aggregate_average.py
----------------------------
Visualization functions for aggregate average experiments.
"""

def plot_error_by_missing_count(x_axis: list, std: list, error: list, run_type: str, save_path=None):
    """
    Plot prediction MAE against ground truth standard deviation for different counts of missing domains.

    Parameters:
        x_axis (list): List of integers representing the number of missing domains.
        std (list): List of floats representing the ground truth standard deviation for each count of missing domains.
        error (list): List of floats representing the prediction MAE for each count of missing domains.
        run_type (str): Type of run, either "repeat" or "nonrepeat".
        save_path (str, optional): Path to save the figure. If None, the figure is not saved.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, std, label="ground truth std", marker="x", color="green")
    plt.plot(x_axis, error, label="Prediction MAE", marker="o", color="blue")
    plt.xticks(x_axis)

    plt.xlabel("Number of Missing Domains")
    plt.ylabel("Error")
    plt.title(f"Prediction MAE and Ground Truth Std. Comparison for {run_type.upper()} Sessions")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()    

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    plt.close()


def plot_average_aggregate_by_missing_count(x_axis: list, avg_dict: dict, std_dict: dict, run_type: str, save_path=None):
    """
    Plot average scores with error bars for different methods against the number of unknown domains.

    Parameters:
        x_axis (list): List of integers representing the number of unknown domains.
        avg_dict (dict): Dictionary with keys representing the different methods ("best", "random", "gt") and values as lists of average scores.
        std_dict (dict): Dictionary with keys representing the different methods ("best", "random", "gt") and values as lists of standard deviations.
        run_type (str): Type of run, either "repeat" or "nonrepeat".
        save_path (str, optional): Path to save the figure. If None, the figure is not saved.
    """
    plt.figure(figsize=(10, 6))

    plt.errorbar(x_axis, avg_dict["best"], yerr=std_dict["best"], label="Best", fmt='-o', capsize=5, color="blue")
    plt.errorbar(x_axis, avg_dict["random"], yerr=std_dict["random"], label="Random", fmt='-o', capsize=5, color="orange")
    plt.errorbar(x_axis, avg_dict["gt"], yerr=std_dict["gt"], label="Ground Truth", fmt='-o', capsize=5, color="green")
    plt.xticks(x_axis)

    # Labels and Title
    plt.xlabel("Number of Unknown Domains")
    
    # Different ylabel and title based on run_type
    if run_type == "repeat":
        plt.ylabel("Average Improvement (Higher is Better)")
        plt.title(f"{run_type.upper()} Session Average Improvement by Number of Unknown Domains")
    else:
        plt.ylabel("Average Score")
        plt.title(f"{run_type.upper()} Session Average Score by Number of Unknown Domains")

    # Legend and Grid
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    # Show plot
    plt.show()
    plt.close()