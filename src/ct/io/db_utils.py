"""
ct.io.db_utils
-----------------
This module contains functions that help with connecting to the database and running SQL queries.
"""

import pandas as pd
from contextlib import contextmanager
from urllib.parse import quote_plus
from sqlalchemy import create_engine, Engine
from sshtunnel import SSHTunnelForwarder
from typing import Iterator

import ct.utils.config_io as config_io

from ct.utils.logger import get_logger
logger = get_logger(__name__)

@contextmanager
def connect_engine(filename: str) -> Iterator[Engine]:
    """
    Open an SSH tunnel and yield a SQLAlchemy Engine suitable for pandas.read_sql.
    Closes the tunnel automatically when the context exits.

    Parameters:
        filename (str): Path to the JSON configuration file containing connection details.
    """
    cfg = config_io.load_json_config(filename)
    mode = cfg.get("connection_mode")

    if mode == "direct":
        logger.info("Creating direct database connection...")
        url = cfg["sqlalchemy_url"]
        engine = create_engine(url, pool_pre_ping=True)
        logger.info("Database engine created.")
        yield engine
        return
    
    elif mode == "ssh_tunnel":
        logger.info(f"Establishing SSH tunnel to {cfg['ssh_address_or_host']}...")
        server = SSHTunnelForwarder(
            ssh_address=(cfg["ssh_address_or_host"], 22),
            ssh_username=cfg["ssh_user"],
            ssh_pkey=cfg["key_path"],
            remote_bind_address=(cfg["host_name"], cfg.get("port", 3306)),
        )
        server.start()
        try:
            user = cfg["username"]
            pw = quote_plus(cfg["password"])  # URL-encode in case of special chars
            host = f"127.0.0.1:{server.local_bind_port}"
            db = cfg.get("database")  # optional; include if present

            url = f"mysql+pymysql://{user}:{pw}@{host}" + (f"/{db}" if db else "")

            engine = create_engine(
                url,
                pool_pre_ping=True,   # helps recover dropped connections
            )
            logger.info("SSH tunnel established and database engine created.")
            yield engine
        finally:
            server.stop()
            
    else:
        logger.error(f"Unknown connection mode: {mode}")
        raise ValueError(f"Unknown connection mode: {mode}")

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
        logger.info("Running SQL query directly.")
        return pd.read_sql(query, con)
    else:
        # if query string is an SQL file, read the file and run the query
        with open(query, "r") as f:
            logger.info(f"Loading SQL query from file: {query}")
            return pd.read_sql(f.read(), con)