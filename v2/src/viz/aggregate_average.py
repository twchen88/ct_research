import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import src.experiments.aggregate_average as core
import src.experiments.file_io as file_io



def plot_error_by_missing_count(x_axis, std, error, run_type: str, save_path=None):
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


def plot_average_aggregate_by_missing_count(x_axis, avg_dict, std_dict, run_type: str, save_path=None):
    plt.figure(figsize=(10, 6))

    plt.errorbar(x_axis, avg_dict["best"], yerr=std_dict["best"], label="Best", fmt='-o', capsize=5, color="blue")
    plt.errorbar(x_axis, avg_dict["random"], yerr=std_dict["random"], label="Random", fmt='-o', capsize=5, color="orange")
    plt.errorbar(x_axis, avg_dict["gt"], yerr=std_dict["gt"], label="Ground Truth", fmt='-o', capsize=5, color="green")
    plt.xticks(x_axis)

    # Labels and Title
    plt.xlabel("Number of Unknown Domains")
    plt.ylabel("Average Score")
    plt.title(f"{run_type.upper()} Session Average Score")

    # Legend and Grid
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    # Show plot
    plt.show()
    plt.close()