import pymysql
import pandas as pd
from sshtunnel import SSHTunnelForwarder
from datetime import datetime

import ct.utils.config_loading as config_loading

"""
src/data/dat_io.py
-----------------
This module contains functions that help with connecting to the database and running SQL queries.
* connect: load connection configuration and connects to the SQL database using SSH tunneling.
* load_sql: load and run a SQL query from sql/*.sql file and returns a pandas DataFrame.
* save_metadata: save metadata about the output file and configuration used to generate it when running a query.
"""

def connect(filename: str) -> pymysql.connections.Connection:
    """
    Takes in the file name of the file that contains credentials and connection information, returns a connection object
    
    Parameters:
        filename (str): The path to the JSON file containing the database connection configuration.

    Returns:
        pymysql.connections.Connection: A connection object to the MySQL database.
    """
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

def load_sql(query: str, con: pymysql.connections.Connection) -> pd.DataFrame:
    """
    Takes in a string of query or an SQL file and connection object, returns a dataframe with read results.
    
    Parameters:
        query (str): The SQL query string or the path to an SQL file.
        con (pymysql.connections.Connection): The connection object to the MySQL database.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the results of the SQL query.
    """
    # if query string is not an SQL file, run the query directly
    if query[-4:] != ".sql":
        return pd.read_sql(query, con)
    else:
        # if query string is an SQL file, read the file and run the query
        with open(query, "r") as f:
            return pd.read_sql(f.read(), con)