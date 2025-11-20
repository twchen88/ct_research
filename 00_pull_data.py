import argparse
import yaml
from datetime import datetime
from pandas import DataFrame

import ct.data.db_utils as db_utils
import ct.data.data_io as data_io
import ct.utils.config_io as config_io
from ct.utils.metadata import get_git_commit_hash
from ct.utils.logger import configure_logging, install_excepthook, get_logger

def save_metadata(output_path: str, config_path: str, config: dict) -> str:
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
    return meta_path

if __name__ == "__main__":
    ## parse command line arguments
    parser = argparse.ArgumentParser(description="Pull data from the database and save it to a CSV file.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration file (YAML format) containing database connection details and query file paths."
        )
    parser.add_argument(
        "--log_file",
        required=True,
        help="Path to the log file where logs will be saved."
        )
    args = parser.parse_args()
    configure_logging(args.log_file)
    install_excepthook()

    ## load configuration file
    config = config_io.load_yaml_config(args.config)
    # check config version
    if config.get("schema_version") == 1:
    # either migrate or error cleanly
        raise ValueError("Config schema_version=1 is no longer supported; please migrate to v2.")
    
    ## set up logging
    logger = get_logger(config.get("logger_name"))
    
    print(f"SQL File Path: {config['source']['sql_file_path']}")
    logger.info(f"SQL File Path: {config['source']['sql_file_path']}")

    # establish connection to the database (via SQLAlchemy Engine)
    try:
        with db_utils.connect_engine(config["source"]["sql_params"]) as engine:
            print("Connection established.")
            logger.info("Database connection established.")

            # load SQL query results from SQL file (or raw SQL string)
            print("Loading SQL query results...")
            data: DataFrame = db_utils.load_sql(config["source"]["sql_file_path"], engine)

    except Exception as e:
        # If anything fails (tunnel, engine, read_sql), surface a clear error
        logger.error(f"Failed to load data from SQL: {e}")
        raise RuntimeError(f"Failed to load data from SQL: {e}") from e

    ## save the query results to a CSV file
    output_data_file_path = config["output"]["dest"] + config["output"]["filename"]
    print(f"Saving query results to {output_data_file_path}...")
    data_io.write_sessions_to_csv(output_data_file_path, data)
    logger.info(f"Query results saved to {output_data_file_path}.")

    ## save metadata about the output file and configuration used to generate it
    meta_path = save_metadata(output_data_file_path, args.config, config)
    logger.info(f"Metadata saved to {meta_path}.")
    
    print("Data pull completed successfully.")
    logger.info("Data pull completed successfully.")