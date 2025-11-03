import pymysql
import pandas as pd
from contextlib import contextmanager
from urllib.parse import quote_plus
from sqlalchemy import create_engine, Engine
from sshtunnel import SSHTunnelForwarder
from typing import Iterator

import ct.utils.config_loading as config_loading

"""
src/data/dat_io.py
-----------------
This module contains functions that help with connecting to the database and running SQL queries.
* connect: load connection configuration and connects to the SQL database using SSH tunneling.
* load_sql: load and run a SQL query from sql/*.sql file and returns a pandas DataFrame.
* save_metadata: save metadata about the output file and configuration used to generate it when running a query.
"""

@contextmanager
def connect_engine(filename: str) -> Iterator[Engine]:
    """
    Open an SSH tunnel and yield a SQLAlchemy Engine suitable for pandas.read_sql.
    Closes the tunnel automatically when the context exits.
    """
    cfg = config_loading.load_json_config(filename)
    print("Connecting to database (via SQLAlchemy)...")

    server = SSHTunnelForwarder(
        ssh_address=(cfg["ssh_address_or_host"], 22),
        ssh_username=cfg["ssh_user"],
        ssh_pkey=cfg["key_path"],
        remote_bind_address=(cfg["host_name"], cfg.get("port", 3306)),
    )
    server.start()
    try:
        # Build a proper SQLAlchemy URL for MySQL+pymysql on the LOCAL forwarded port
        user = cfg["username"]
        pw = quote_plus(cfg["password"])  # URL-encode in case of special chars
        host = f"127.0.0.1:{server.local_bind_port}"
        db = cfg.get("database")  # optional; include if present

        url = f"mysql+pymysql://{user}:{pw}@{host}" + (f"/{db}" if db else "")

        engine = create_engine(
            url,
            pool_pre_ping=True,   # helps recover dropped connections
        )
        print("Connection successful (engine ready).")
        yield engine
    finally:
        server.stop()

def load_sql(query: str, con: Engine) -> pd.DataFrame:
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