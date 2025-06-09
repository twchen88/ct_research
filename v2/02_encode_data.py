import argparse
import pandas as pd
import numpy as np

import src.utils.config_loading as config_loading
import src.data.data_io as data_io
import src.data.encoding as encoding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode data so that it is compatible for input to the prediction neural network.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file with source and target paths."
    )
    args = parser.parse_args()

    config = config_loading.load_yaml_config(args.config)
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

    # Combine with remaining data
    print("Combining encoded data with encoding and target columns...")
    encoded_data = np.hstack((
        df[encoding_columns].to_numpy(),
        encoded_data,
        df[target_columns].to_numpy()
    ))
    print(f"Encoded data shape: {encoded_data.shape}")

    # Save encoded data to CSV file
    print(f"Saving encoded data to {target_path}...")
    data_io.write_sessions_to_npy(target_path, encoded_data)

    # Save metadata about the output file and configuration used to generate it
    print("Saving metadata...")
    encoding.save_metadata(
        input_path=source_path,
        output_path=target_path,
        config_path=args.config,
        column_names= df.columns.tolist()
    )