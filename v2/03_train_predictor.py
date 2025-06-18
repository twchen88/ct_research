import argparse
import torch
import matplotlib.pyplot as plt

import src.training.file_io as file_io
import src.training.training_torch as training
import src.training.evaluation_torch as evaluation
import src.utils.config_loading as config_loading
import src.viz.training as visualize

from datetime import datetime
from pathlib import Path

from src.shared.model_torch import Predictor
from src.utils.reproducibility import set_global_seed
from src.utils.metadata import get_git_commit_hash


def create_run_dir(base_dir: str, tag: str):
    timestamp = datetime.now().strftime("%Y%m%d")
    tag_part = f"_{tag}"
    run_id = f"run_{timestamp}{tag_part}"
    run_path = Path(base_dir) / run_id
    run_path.mkdir(parents=True, exist_ok=False)
    return run_id, run_path


if __name__ == "__main__":
    # load config
    parser = argparse.ArgumentParser(description="Train a model with the given configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--tag", type=str, required=True, help="Tag to identify the run, describe the experiment in a few words.")
    args = parser.parse_args()
    config = config_loading.load_yaml_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # set variables and other misc settings like device
    device = config["settings"]["device"]
    run_desc = config["settings"]["run_desc"]
    seed = config["settings"]["seed"]
    ratio_test = config["settings"]["test_ratio"]
    ratio_validation = config["settings"]["val_ratio"]
    
    data_source = config["data"]["source"]
    destination_base = config["data"]["destination_base"]

    optimizer_name = config["hyperparams"]["optimizer"]
    loss_function_name = config["hyperparams"]["loss_function"]
    epochs = config["hyperparams"]["epochs"]
    batch_size = config["hyperparams"]["batch_size"]
    n_samples = config["hyperparams"]["n_samples"]
    learning_rate = config["hyperparams"]["learning_rate"]
    dims = config["hyperparams"]["dims"]

    # get git commit hash
    git_commit_hash = get_git_commit_hash()
    print(f"Git commit hash: {git_commit_hash}")
    
    # set global seed and device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is selected as device, but no GPU is available. Please select 'cpu' as device.")
    elif device == "cuda":
        device = "cuda:0"
    else:
        device = "cpu"
    set_global_seed(seed)

    # set output destination
    output_destination = create_run_dir(base_dir=destination_base, tag=args.tag)[1]

    ## data preparation
    # load data
    data = file_io.load_data(data_source)

    # split data into train, validatio, and test sets
    train_data, test_data = training.split_train_test(data, ratio=ratio_test, n_samples=n_samples)
    train_data, valid_data = training.split_train_test(train_data, ratio=ratio_validation, n_samples=n_samples)
    # split input and target data, turn into tensors
    train_x, train_y = training.split_input_target(train_data, dims=dims)
    valid_x, valid_y = training.split_input_target(valid_data, dims=dims)
    test_x, test_y = training.split_input_target(test_data, dims=dims)
    # create datasets and dataloaders
    train_dataset = training.custom_dataset(train_x, train_y)
    train_data_loader = training.get_dataloader(train_dataset, batch_size, suffle=True)
    valid_dataset = training.custom_dataset(valid_x, valid_y)
    valid_data_loader = training.get_dataloader(valid_dataset, batch_size, suffle=False)
    test_dataset = training.custom_dataset(test_x, test_y)
    test_data_loader = training.get_dataloader(test_dataset, batch_size, suffle=False)

    ## training
    model = Predictor().to(device)
    optimizer = training.get_optimizer(optimizer_name, learning_rate, model)
    loss_function = training.get_loss_function(loss_function_name)
    train_loss_history, val_loss_history = training.train_model(
                                                    model=model,
                                                    train_data_loader=train_data_loader,
                                                    val_data_loader=valid_data_loader,
                                                    epochs=epochs,
                                                    optimizer=optimizer,
                                                    loss_function=loss_function,
                                                    device=device
                                                )

    ## basic evaluation: evaluate loss and error on test set
    test_loss = training.evaluate_loss(model, test_data_loader, loss_function)
    predictions = evaluation.predict(model, test_data_loader)
    test_error = evaluation.evaluate_error(predictions, test_y)

    ## plotting
    curve_plot = plt.figure(figsize=(10, 6))
    visualize.plot_single_curve(curve_plot, train_loss_history, val_loss_history, 
                                output_path=f"{output_destination}/loss_curve.png")

    # save model
    file_io.save_model(model, f"{output_destination}/model.pt")

    # copy config file
    file_io.copy_config_file(args.config, f"{output_destination}/config.yaml")

    # save metrics
    metrics = {
        "test_loss": test_loss,
        "test_error": test_error.item(),  # Convert tensor to scalar
    }
    file_io.save_metrics(metrics, f"{output_destination}/metrics.yaml")

    # save results
    results = {
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "test_set": test_data,
        "test_predictions": predictions.numpy()  # Convert tensor to numpy array
    }
    file_io.save_metrics(results, f"{output_destination}/results.npz")

    # save metadata
    file_io.save_metadata(
        run_desc=run_desc,
        git_commit_hash=git_commit_hash,
        input_path=data_source,
        output_path=f"{output_destination}",
        config_path=f"{output_destination}/config.yaml",
        metrics_path=f"{output_destination}/results.npz",
        plots_path=f"{output_destination}/loss_curve.png",
    )