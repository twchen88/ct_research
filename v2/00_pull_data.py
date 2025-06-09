## import libraries
import argparse
## import custom modules
import src.data.db_utils as db_utils
import src.data.data_io as data_io
import src.utils.config_loading as config_loading

if __name__ == "__main__":
    ## parse command line arguments
    parser = argparse.ArgumentParser(description="Pull data from the database and save it to a CSV file.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration file (YAML format) containing database connection details and query file paths."
        )
    args = parser.parse_args()

    ## load configuration file
    config = config_loading.load_yaml_config(args.config)
    print(f"SQL File Path: {config['source']['sql_file_path']}")

    ## establish connection to the database
    con = None
    try:
        con = db_utils.connect(config["source"]["sql_params"])
        print("Connection established.")
    except Exception as e:
        print(f"Error connecting to the database: {e}")

    ## load SQL query results from SQL file
    print("Loading SQL query results...")
    if con is None:
        raise Exception("Database connection could not be established. Exiting.")
    data = db_utils.load_sql(config["source"]["sql_file_path"], con)

    ## save the query results to a CSV file
    output_data_file_path = config["output"]["dest"] + config["output"]["filename"]
    print(f"Saving query results to {output_data_file_path}...")
    data_io.write_sessions_to_csv(output_data_file_path, data)

    ## save metadata about the output file and configuration used to generate it
    db_utils.save_metadata(output_data_file_path, args.config, config)

    ## close the database connection
    try:
        if con is not None:
            con.close()
            print("Connection closed.")
    except Exception as e:
        print(f"Error closing the database connection: {e}")
    print("Data pull completed successfully.")