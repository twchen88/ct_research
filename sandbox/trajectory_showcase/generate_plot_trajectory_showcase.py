import matplotlib.pyplot as plt
from typing import List, Optional
from pathlib import Path
import argparse
import numpy as np
from pathlib import Path

import ct.utils.config_io as config_io
import  ct.experiments.trajectory as core

from ct.experiments.shared import load_model
from ct.utils.reproducibility import set_global_seed

def compare_trajectories(
    best_performance: List[float],
    random_performance: List[float],
    worst_performance: List[float],
    esp_performance: List[float],
    save_path: Optional[Path] = None,
    dpi: int = 300
):
    """
    Stolen from src/viz/trajectory.py and slightly modified to add in ESP mode.
    Compare the performance of three different modes over time and optionally save the plot.

    Parameters:
        best_performance (List[float]): List of performance values for the best mode.
        random_performance (List[float]): List of performance values for the random mode.
        worst_performance (List[float]): List of performance values for the worst mode.
        esp_performance (List[float]): List of performance values for the ESP mode.
        save_path (Optional[Path]): Path to save the plot image (e.g., 'results/trajectory_plot.png'). If None, the plot is only shown.
        dpi (int): Dots per inch for saving the figure (default: 300).
    """
    
    plt.figure(figsize=(8, 5))
    x_values = range(1, 15)
    plt.plot(x_values, best_performance, label="best", marker="o")
    plt.plot(x_values, random_performance, label="random", marker="o")
    plt.plot(x_values, worst_performance, label="worst", marker="o")
    plt.plot(x_values, esp_performance, label="ESP", marker="o")

    plt.xlabel("Timestep")
    plt.ylabel("Known Domain Average")
    plt.title("Known Domain Average Over Time")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close()


if __name__ == '__main__':
    # load config (default to the previous non percentile one)
    parser = argparse.ArgumentParser(description="Train a model with the given configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    config = config_io.load_yaml_config(args.config)
    print(f"Loaded configuration from {args.config}")

    ## set config variables
    output_base = config["settings"]["output_base"]
    model_path = config["settings"]["model_path"]
    device = config["settings"]["device"]
    seed = config["settings"]["seed"]

    # set global seed and device
    set_global_seed(seed)

    ## load model
    model = load_model(model_path, device)

    ## simulate trajectories
    # best
    best_peformance, best_final_scores, best_order = core.trajectory(model, "best")
    # random
    random_peformance, random_final_scores, random_order = core.trajectory(model, "middle")
    # worst
    worst_peformance, worst_final_scores, worst_order = core.trajectory(model, "worst")
    # esp showcase, add noise to best performance as a placeholder
    esp_peformance = [score + np.random.normal(0, 0.005) for score in best_peformance]

    ## visualize and save results in output directory
    figure_save_path = Path("sandbox/20251028/trajectory_plot.png")
    compare_trajectories(
        best_peformance,
        random_peformance,
        worst_peformance,
        esp_peformance,  # Placeholder for ESP performance
        save_path=figure_save_path
    )