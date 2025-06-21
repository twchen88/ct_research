import matplotlib.pyplot as plt
from typing import List, Optional
from pathlib import Path

def compare_trajectories(
    best_peformance: List[float],
    middle_peformance: List[float],
    worst_peformance: List[float],
    save_path: Optional[Path] = None,
    dpi: int = 300
):
    """
    Compare the performance of three different modes over time and optionally save the plot.

    Parameters:
    - best_peformance: List of performance values for the best mode.
    - middle_peformance: List of performance values for the middle mode.
    - worst_peformance: List of performance values for the worst mode.
    - save_path: Path to save the plot image (e.g., 'results/trajectory_plot.png'). If None, the plot is only shown.
    - dpi: Dots per inch for saving the figure (default: 300).
    """
    
    plt.figure(figsize=(8, 5))
    x_values = range(1, 15)
    plt.plot(x_values, best_peformance, label="best", marker="o")
    plt.plot(x_values, middle_peformance, label="random", marker="o")
    plt.plot(x_values, worst_peformance, label="worst", marker="o")

    plt.xlabel("Timestep")
    plt.ylabel("Known Domain Average")
    plt.title("Known Domain Average Over Time for Three Different Modes")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close()