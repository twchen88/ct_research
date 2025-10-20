## import libraries
import argparse
import yaml
from datetime import datetime
## import custom modules
import ct.data.db_utils as db_utils
import ct.data.data_io as data_io
import ct.utils.config_loading as config_loading
from ct.utils.metadata import get_git_commit_hash

def save_metadata(output_path: str, config_path: str, config: dict) -> None:
    """
    Saves metadata about the output file and configuration used to generate it.
    Parameters:
        output_path (str): The path to the output file.
        config_path (str): The path to the configuration file used to generate the output.
        config (dict): The configuration dictionary containing source and database information.
    """
    metadata = {
        "output_file": output_path,
        "config_file": config_path,
        "timestamp": datetime.now().isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "query": config['source']['sql_file_path'],
        "database": config['source']['sql_params'],
    }
    meta_path = output_path.replace(".csv", ".meta.yaml")
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f)

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
    save_metadata(output_data_file_path, args.config, config)

    ## close the database connection
    try:
        if con is not None:
            con.close()
            print("Connection closed.")
    except Exception as e:
        print(f"Error closing the database connection: {e}")
    print("Data pull completed successfully.")