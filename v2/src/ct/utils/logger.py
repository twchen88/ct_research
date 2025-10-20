import logging
from pathlib import Path

"""
src/utils/logger.py
--------------------------------
This module provides a utility function to create and configure loggers.
"""

def get_logger(name: str, log_file=None) -> logging.Logger:
    """
    Returns a logger with the specified name.
    
    Parameters:
        name (str): The name of the logger.
        
    Returns:
        logging.Logger: A logger instance with the specified name.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Optional file output
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter("[%(levelname)s] %(asctime)s — %(message)s")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    return logger