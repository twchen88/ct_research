import argparse
import gc
import yaml
import pandas as pd

import ct.data.data_io as data_io
import ct.data.preprocessing as preprocessing
import ct.utils.config_loading as config_loading

from datetime import datetime
from typing import Dict, Any


def save_metadata(input_path : str, output_path : str, config_path : str, config : Dict[str, Any], stats : Dict[Any, Any]) -> None:
    """
    Saves metadata about the preprocessing operation, including input and output file paths, configuration used, timestamp, and statistics.
    
    Parameters:
        input_path (str): Path to the input file.
        output_path (str): Path to the output file.
        config_path (str): Path to the configuration file used for preprocessing.
        config (dict): Configuration dictionary containing parameters used in preprocessing.
        stats (dict): Dictionary containing statistics about the output file, such as number of rows and columns.
    """
    metadata = {
        "input_file": input_path,
        "output_file": output_path,
        "config_file": config_path,
        "timestamp": datetime.now().isoformat(),
        "params": config["filter_params"],
        "output_file_stats": stats
    }
    meta_path = output_path.replace(".csv", ".meta.yaml")
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamed preprocessing for large session data.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file with source and target paths."
    )
    args = parser.parse_args()

    config = config_loading.load_yaml_config(args.config)
    # check config version
    if config.get("schema_version") == 1:
    # either migrate or error cleanly
        raise ValueError("Config schema_version=1 is no longer supported; please migrate to v2.")

    input_path = config['source']['directory'] + config['source']['filename']
    output_path = config['target']['directory'] + config['target']['filename']
    usage_frequency_threshold = config['filter_params']['usage_frequency']
    usage_days_threshold = config['filter_params']['usage_days']
    eps_days = config['filter_params']['eps_days']
    min_samples = config['filter_params']['min_samples']

    print(f"Streaming and processing raw data from {input_path}")
    processed_chunks = []
    user_usage_tracker = {}

    # First pass: collect usage stats
    for chunk in data_io.read_raw_session_chunks(input_path):
        chunk = preprocessing.drop_duplicates(chunk, based_on=['id'])
        chunk = preprocessing.filter_datetime_outliers(chunk, eps_days=eps_days, min_samples=min_samples)
        
        chunk['start_date'] = chunk['start_time'].dt.date
        for pid, group in chunk.groupby('patient_id'):
            if pid not in user_usage_tracker:
                user_usage_tracker[pid] = {
                    'min': group['start_time'].min(),
                    'max': group['start_time'].max(),
                    'days': set(group['start_date'])
                }
            else:
                user_usage_tracker[pid]['min'] = min(user_usage_tracker[pid]['min'], group['start_time'].min())
                user_usage_tracker[pid]['max'] = max(user_usage_tracker[pid]['max'], group['start_time'].max())
                user_usage_tracker[pid]['days'].update(group['start_date'])

        del chunk
        gc.collect()

    # Compute final usage stats
    usage_summary = []
    for pid, stats in user_usage_tracker.items():
        usage_time = (stats['max'] - stats['min']).days + 1
        usage_freq = len(stats['days']) / usage_time if usage_time > 0 else 0
        usage_summary.append((pid, len(stats['days']), usage_time, usage_freq))

    usage_df = pd.DataFrame(usage_summary, columns=['patient_id', 'unique_days', 'usage_time', 'usage_freq'])
    passing_users = usage_df[(usage_df['usage_freq'] > usage_frequency_threshold) & (usage_df['usage_time'] > usage_days_threshold)]['patient_id'].tolist()
    passing_users = set(passing_users)

    # Second pass: process only passing users
    all_sessions = []
    for i, chunk in enumerate(data_io.read_raw_session_chunks(input_path)):
        chunk = preprocessing.drop_duplicates(chunk, based_on=['id'])
        chunk = preprocessing.filter_datetime_outliers(chunk, eps_days=eps_days, min_samples=min_samples)
        chunk = chunk[chunk['patient_id'].isin(passing_users)]

        for pid, group in chunk.groupby("patient_id"):
            processed = preprocessing.extract_session_data(group)
            all_sessions.append(processed)

        print(f"Processed chunk {i}, rows: {len(chunk)}")
        del chunk
        gc.collect()
    

    final_df = pd.concat(all_sessions, ignore_index=True)
    stats = {
        'num_users': final_df['patient_id'].nunique(),
        'num_sessions': final_df.shape[0]
    }

    final_df = preprocessing.convert_to_percentile(final_df)

    print(f"Saving processed data and metadata to {output_path}")
    data_io.write_sessions_to_csv(output_path, final_df)
    save_metadata(
        input_path=input_path,
        output_path=output_path,
        config_path=args.config,
        config=config,
        stats=stats
    )