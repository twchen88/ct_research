import logging

def get_logger(name: str) -> logging.Logger:
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
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger