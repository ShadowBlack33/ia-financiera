# logger.py
import logging
import os

def setup_logger(log_file='project.log'):
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger('financial_project')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(os.path.join("logs", log_file))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

logger = setup_logger()


