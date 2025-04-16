import logging
from pathlib import Path
from datetime import datetime


# Configure the root logger
def setup_logger(log_folder_name:str, file_name:str=None):
    """
    Function to set up logger

    Args:
        log_folder_name (str): name of subfolder directoru under logs/
        file_name (str): Optional, default is None. Prefix name of log file under subfolder. <filename>_<timestamp>.log
    """
    # Create a logs directory if it doesn't exist
    log_dir = Path(__file__).parents[2] / 'logs' / log_folder_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a timestamp for the log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS

    if file_name:
        log_file_name = f"{file_name}_{timestamp}.log"
    else:
        log_file_name = f"{timestamp}.log"

    log_file = log_dir / log_file_name  # Set the log file name

    logging.basicConfig(
        level=logging.INFO,  # Set the minimum logging level
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ]
    )
    logging.info("Logger is set up.")