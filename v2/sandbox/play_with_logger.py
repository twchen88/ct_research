import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.logger import get_logger

if __name__ == "__main__":
    log = get_logger("play_with_logger")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
    log.debug("This is a debug message.")
    
    # Check if the logger has handlers
    if not log.hasHandlers():
        log.info("Logger has no handlers.")
    else:
        log.info("Logger has handlers.")