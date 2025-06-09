### import libraries
import argparse
import src.data.data_io as data_io
import src.data.preprocessing as preprocessing
import src.utils.config_loading as config_loading

if __name__ == "__main__":
    ## parse commmand line arguments
    parser = argparse.ArgumentParser(description="Preprocess data from a CSV file and save it to a new CSV file.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration file (YAML format) containing input and output file paths."
    )
    args = parser.parse_args()

    ## load configuration file
    print(f"Loading configuration from {args.config}")
    config = config_loading.load_yaml_config(args.config)
    # extract information from the configuration file
    input_path = config['source']['directory'] + config['source']['filename']
    output_path = config['target']['directory'] + config['target']['filename']

    usage_frequency_threshold = config['filter_params']['usage_frequency']
    usage_days_threshold = config['filter_params']['usage_days']

    eps_days = config['filter_params']['eps_days']
    min_samples = config['filter_params']['min_samples']
    ## debug prints
    print("==========================================")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Usage frequency threshold: {usage_frequency_threshold}")
    print(f"Usage days threshold: {usage_days_threshold}")
    print(f"DBSCAN eps_days: {eps_days}")
    print(f"DBSCAN min_samples: {min_samples}")
    print("==========================================")

    ## read data from data/raw_data/
    print(f"Reading raw data from {input_path}")
    raw_data = data_io.read_raw_session_file(input_path)

    print(f"Processing raw data with {len(raw_data)} sessions")
    ## drop duplicate sessions based on 'id' column (id == session id)
    raw_data_dropped = preprocessing.drop_duplicates(raw_data, based_on=['id'])

    ## filter out session gaps using DBSCAN clustering
    raw_data_filtered = preprocessing.filter_datetime_outliers(raw_data_dropped, column='start_time', eps_days=eps_days, min_samples=min_samples)

    ## find usage frequency of each user
    usage_df = preprocessing.find_usage_frequency(raw_data_filtered)

    ## filter out users with low usage frequency
    # find users that have usage frequency >= usage_frequency and usage_time >= usage_days
    usage_df_filtered = usage_df[(usage_df['usage_freq'] > usage_frequency_threshold) & (usage_df['usage_time'] > usage_days_threshold)]
    # get patient ids of users that pass the filter
    patient_ids = usage_df_filtered['patient_id'].tolist()
    # filter the raw data filtered to only include sessions of these patients
    filtered_data = raw_data_filtered[raw_data_filtered['patient_id'].isin(patient_ids)]

    ## extract session data, grouping by patient and sorted by start time, this now returns a dataframe where each row contains past information as well
    session_data = filtered_data.groupby('patient_id')[filtered_data.columns].apply(preprocessing.extract_session_data).reset_index(drop=True)

    ## find statistics of the processed data
    stats = {
        'num_users': session_data['patient_id'].nunique(),
        'num_sessions': session_data.shape[0]
    }

    print(f"Saving processed data to and metadata to {output_path}")
    ## save the processed data to a CSV file
    data_io.write_sessions_to_csv(output_path, session_data)

    ## save metadata to a YAML file
    preprocessing.save_metadata(
        input_path=input_path,
        output_path=output_path,
        config_path=args.config,
        config=config,
        stats=stats
    )