import argparse
import yaml
import numpy as np

import ct.experiments.file_io as file_io
import ct.experiments.aggregate_average as core
import ct.experiments.shared as shared
import ct.viz.aggregate_average as viz

from pathlib import Path
from datetime import datetime

from ct.utils.reproducibility import set_global_seed
from ct.utils.metadata import get_git_commit_hash
from ct.utils.config_loading import load_yaml_config

def create_run_dir(base_dir: str, tag: str, run_type: str):
    timestamp = datetime.now().strftime("%Y%m%d")
    tag_part = f"_{tag}"
    run_type_part = f"_{run_type}"
    run_id = f"run_{timestamp}{tag_part}{run_type_part}"
    run_path = Path(base_dir) / run_id
    run_path.mkdir(parents=True, exist_ok=False)
    return run_id, run_path


def save_metadata(output_path, config, git_commit_hash, figure_paths):
    metadata = {
        "config": config,
        "git_commit_hash": git_commit_hash,
        "figure_path": figure_paths
    }
    metadata_path = output_path / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_path}")

def accuracy_assessment_pipeline(test_x, test_y, test_predictions, output_destination, figure_path, run_type):
    missing_counts, mean_errors_list, ground_truth_std_list, ground_truth_dict = core.evaluate_error_by_missing_count(
        test_x=test_x,
        test_y=test_y,
        test_predictions=test_predictions,
        dims=14
    )

    # save ground truth matrices for each missing count
    file_io.save_ground_truth_matrices(ground_truth_dict, output_destination / "ground_truth_matrices.npz")

    # plot error by missing count and save figure
    viz.plot_error_by_missing_count(
        x_axis=missing_counts,
        std=ground_truth_std_list,
        error=mean_errors_list,
        save_path=output_destination / figure_path,
        run_type=run_type
    )

def aggregate_average_pipeline(test_x, test_y, model, figure_path, run_type):
    # split encoding and scores
    gt_encoding, cur_score_gt = core.split_encoding_and_scores(test_x, dims=14)

    cur_score_gt_decoded = core.decode_missing_indicator(cur_score_gt)
    future_score_gt = test_y
    # define missing counts
    missing_counts = list(range(0, 14))

    # ground truth stats
    avg_lst_gt, std_lst_gt = core.average_scores_by_missing_counts(missing_counts, cur_score_gt_decoded, future_score_gt, gt_encoding)

    # model predictions
    valid_mask = core.find_valid_domains(cur_score_gt_decoded, run_type=run_type)

    (best_enc, best_pred), (rand_enc, rand_pred) = core.choose_best_and_random(model, cur_score_gt, missing_counts, valid_mask)
    avg_lst_best, std_lst_best = core.average_scores_by_missing_counts(missing_counts, cur_score_gt_decoded, best_pred, best_enc)
    avg_lst_rand, std_lst_rand = core.average_scores_by_missing_counts(missing_counts, cur_score_gt_decoded, rand_pred, rand_enc)

    avg_dict = {
        "best": avg_lst_best,
        "random": avg_lst_rand,
        "gt": avg_lst_gt
    }

    std_dict = {
        "best": std_lst_best,
        "random": std_lst_rand,
        "gt": std_lst_gt
    }

    if run_type == "nonrepeat":
        for name, arr in {"best": avg_lst_best, "random": avg_lst_rand, "gt": avg_lst_gt}.items():
            arr_np = np.asarray(arr)
            if (arr_np < 0).any():
                idxs = np.where(arr_np < 0)[0].tolist()
                raise AssertionError(f"Nonrepeat {name} averages contained negatives at missing_counts indices {idxs}: {arr_np[idxs]}")

    # plot aggregate average and save figure
    viz.plot_average_aggregate_by_missing_count(missing_counts, avg_dict, std_dict, run_type, figure_path)



if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser(description="Aggregate average scores from multiple runs for a more quantitative view.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--tag", type=str, required=True, help="Tag to identify the run, describe the experiment in a few words.")
    parser.add_argument("--run_type", type=str, required=True, help="repeat or nonrepeat")
    args = parser.parse_args()

    # load configuration
    config = load_yaml_config(args.config)
    run_type = args.run_type # repeat vs non-repeat
    device = config["settings"]["device"] # cpu or cuda
    data_source = config["data"]["data_source"] # npz file with multiple arrays
    model_source = config["data"]["model_source"] # path to the model
    destination_base = config["data"]["destination_base"]
    seed = config["settings"]["seed"]

    ## set output directory
    _, output_destination = create_run_dir(destination_base, args.tag, run_type)

    ## set global seed
    set_global_seed(seed)

    ## get git commit hash
    git_commit_hash = get_git_commit_hash()

    ## read data
    test_x, test_y = file_io.load_test_data(
        file_path=data_source,
        file_names=["test_x", "test_y"]
    )
    # filter for sessions with tasks that are only in one domain
    test_x, sum_mask = core.filter_rows_by_sum(test_x, slice(0, 14), 1)

    test_predictions = file_io.load_saved_test_predictions(
        file_path=data_source
    )

    test_y = test_y[sum_mask]
    test_predictions = test_predictions[sum_mask]

    ## filter by session type
    repeat_mask = core.assign_repeat(test_x)

    # right after parsing args
    run_type_raw = args.run_type
    run_type = run_type_raw.strip().lower().replace("-", "")
    if run_type not in {"repeat", "nonrepeat"}:
        raise ValueError("run_type must be 'repeat' or 'nonrepeat'")

    is_repeat = (run_type == "repeat")

    # if run type is repeat, filter for only repeat sessions
    # if run type is non-repeat, filter for only non-repeat sessions
    mask = repeat_mask if is_repeat else ~repeat_mask

    test_x = test_x[mask]
    test_y = test_y[mask]
    test_predictions = test_predictions[mask]

    ## general setup
    figure_names = ["accuracy_assessment.png", "aggregate_average.png"]

    ## (1) find ground truth std and prediction MAE
    accuracy_assessment_pipeline(test_x, test_y, test_predictions, output_destination, figure_names[0], run_type)

    ## (2) predict scores based on strategy
    # load model
    model = shared.load_model(model_path=model_source, device=device)

    aggregate_average_figure_path = output_destination / figure_names[1]
    
    # run aggregate average pipeline
    aggregate_average_pipeline(
        test_x=test_x,
        test_y=test_y,
        model=model,
        figure_path=aggregate_average_figure_path,
        run_type=run_type)

    # save metadata
    save_metadata(
        output_path=output_destination,
        config=config,
        git_commit_hash=git_commit_hash,
        figure_paths=figure_names
    )