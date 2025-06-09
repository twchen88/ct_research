### Functions that help with connecting to the database and running SQL queries
import pymysql
import yaml
import pandas as pd
from sshtunnel import SSHTunnelForwarder
from datetime import datetime

import src.utils.config_loading as config_loading

# takes in the file name of the file that contains credentials and connection information, returns a connection object
def connect(filename : str) -> pymysql.connections.Connection:
    # load in credential file
    config_dict = config_loading.load_json_config(filename)
    print("Connecting to database...")

    ssh_address = config_dict["ssh_address"]
    ssh_user = config_dict["ssh_user"]
    key_path = config_dict["key_path"]
    host_name = config_dict["host_name"]
    user = config_dict["username"]
    pw = config_dict["password"]

    # port forwarding
    server = SSHTunnelForwarder(
    ssh_address=(ssh_address, 22),
    ssh_username=ssh_user,
    ssh_pkey=key_path,
    remote_bind_address=(host_name, 3306)
    )

    server.start()

    # connection to MySQL
    con = pymysql.connect(user=user, passwd=pw, host='127.0.0.1', port=server.local_bind_port)

    print("Connection Successful")

    return con

# takes in a string of query or an SQL file and connection object, returns a dataframe with read results
def load_sql(query : str, con : pymysql.connections.Connection) -> pd.DataFrame:
    # if query string is not an SQL file, run the query directly
    if query[-4:] != ".sql":
        return pd.read_sql(query, con)
    else:
        # if query string is an SQL file, read the file and run the query
        with open(query, "r") as f:
            return pd.read_sql(f.read(), con)
        
# save metadata about the output file and configuration used to generate it
def save_metadata(output_path : str, config_path : str, config : dict) -> None:
    metadata = {
        "output_file": output_path,
        "config_file": config_path,
        "timestamp": datetime.now().isoformat(),
        "query": config['source']['sql_file_path'],
        "database": config['source']['sql_params'],
    }
    meta_path = output_path.replace(".csv", ".meta.yaml")
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f)