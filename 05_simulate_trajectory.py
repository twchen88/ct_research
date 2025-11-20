import argparse
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path

import ct.utils.config_io as config_io
import  ct.experiments.trajectory as core
import ct.viz.trajectory as viz

from ct.experiments.shared import load_model
from ct.utils.reproducibility import set_global_seed
from ct.utils.metadata import get_git_commit_hash

def create_run_dir(base_dir: str, tag: str):
    timestamp = datetime.now().strftime("%Y%m%d")
    tag_part = f"_{tag}"
    run_id = f"run_{timestamp}{tag_part}"
    run_path = Path(base_dir) / run_id
    run_path.mkdir(parents=True, exist_ok=False)
    return run_id, run_path


def save_metadata(output_path, config, git_commit_hash, figure_path):
    metadata = {
        "config": config,
        "git_commit_hash": git_commit_hash,
        "figure_path": figure_path
    }
    metadata_path = output_path / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_path}")


if __name__ == '__main__':
    # load config
    parser = argparse.ArgumentParser(description="Train a model with the given configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--tag", type=str, required=True, help="Tag to identify the run, describe the experiment in a few words.")
    args = parser.parse_args()
    config = config_io.load_yaml_config(args.config)
    print(f"Loaded configuration from {args.config}")
    # check config version
    if config.get("schema_version") == 1:
    # either migrate or error cleanly
        raise ValueError("Config schema_version=1 is no longer supported; please migrate to v2.")

    ## set config variables
    output_base = config["settings"]["output_base"]
    model_path = config["settings"]["model_path"]
    device = config["settings"]["device"]
    seed = config["settings"]["seed"]

    # get git commit hash
    git_commit_hash = get_git_commit_hash()
    print(f"Git commit hash: {git_commit_hash}")

    # set global seed and device
    set_global_seed(seed)

    # set output directory
    output_destination = create_run_dir(output_base, args.tag)[1]

    ## load model
    model = load_model(model_path, device)

    ## simulate trajectories
    # best
    best_peformance, best_final_scores, best_order = core.trajectory(model, "best")
    # random
    random_peformance, random_final_scores, random_order = core.trajectory(model, "middle")
    # worst
    worst_peformance, worst_final_scores, worst_order = core.trajectory(model, "worst")

    ## visualize and save results in output directory
    figure_save_path = output_destination / "trajectory_plot.png"
    viz.compare_trajectories(
        best_peformance,
        random_peformance,
        worst_peformance,
        save_path=figure_save_path
    )

    ## save metadata
    save_metadata(output_destination, config, git_commit_hash, figure_save_path)