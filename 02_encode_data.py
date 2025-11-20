import argparse
import yaml
import pandas as pd
import numpy as np

import ct.utils.config_io as config_io
import ct.data.data_io as data_io
import ct.data.encoding as encoding

from datetime import datetime
from ct.utils.metadata import get_git_commit_hash


def save_metadata(input_path: str, output_path: str, config_path: str, column_names: list) -> None:
    """
    Saves metadata about the output file and configuration used to generate it.

    Parameters:
        output_file_path (str): Path to the output file.
        config_file_path (str): Path to the configuration file.
        config (dict): Configuration dictionary.
        column_names (list): List of column names in the output file.
    """
    metadata = {
        "time_stamp": datetime.now().isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "input_path": input_path,
        "output_file_path": output_path,
        "config_file_path": config_path,
        "column_names": column_names
    }

    meta_path = output_path.replace(".npy", ".meta.yaml")
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode data so that it is compatible for input to the prediction neural network.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file with source and target paths."
    )
    args = parser.parse_args()

    config = config_io.load_yaml_config(args.config)
    # check config version
    if config.get("schema_version") == 1:
    # either migrate or error cleanly
        raise ValueError("Config schema_version=1 is no longer supported; please migrate to v2.")

    source_path = config['source_path']
    target_path = config['target_path']
    print(f"Source path: {source_path}")
    print(f"Target path: {target_path}")

    # Read preprocessed session data
    print("Reading preprocessed session data...")
    df = data_io.read_preprocessed_session_file(source_path)
    print(f"data dtypes:\n{df.dtypes}")

    # Strip unused columns
    print(f"Data shape before stripping unused columns: {df.shape}")
    print("Stripping unused columns...")
    df = df.drop(columns=["patient_id", "start_time", "time_stamp"])
    print(f"Data shape after stripping unused columns: {df.shape}")

    # Identify three sections of columns
    score_columns = [col for col in df.columns if col.endswith("score")]
    encoding_columns = [col for col in df.columns if col.endswith("encoding")]
    target_columns = [col for col in df.columns if col.endswith("target")]
    print(f"Score columns: {score_columns}")
    print(f"Encoding columns: {encoding_columns}")
    print(f"Target columns: {target_columns}")
    
    # Encode data with missing value indicator (See Dissertation Proposal Section 4.2)
    print("Encoding data with missing value indicator...")
    encoded_data = encoding.create_missing_indicator(df[score_columns].to_numpy(), rand_seed=42)

    # Encode target data
    print("Encoding target data...")
    encoded_target = encoding.encode_target_data(
        df[target_columns].to_numpy(),
        df[encoding_columns].to_numpy()
    )

    # Combine with remaining data
    print("Combining encoded data with encoding and target columns...")
    encoded_data = np.hstack((
        df[encoding_columns].to_numpy(),
        encoded_data,
        encoded_target
    ))
    print(f"Encoded data shape: {encoded_data.shape}")

    # Save encoded data to CSV file
    print(f"Saving encoded data to {target_path}...")
    data_io.write_sessions_to_npy(target_path, encoded_data)

    # Save metadata about the output file and configuration used to generate it
    print("Saving metadata...")
    save_metadata(
        input_path=source_path,
        output_path=target_path,
        config_path=args.config,
        column_names= df.columns.tolist()
    )