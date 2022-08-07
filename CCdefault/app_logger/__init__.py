import logging as lg
from datetime import datetime
import os

ROOT_DIR = os.getcwd()
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d')}"
LOG_DIR = os.path.join(ROOT_DIR, "app_logs")


def App_Logger(logger_name):
    file_name = f"{CURRENT_TIME_STAMP}_{logger_name}_logs.log"
    log_file = os.path.join(LOG_DIR, file_name)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = lg.getLogger(logger_name)
    logger.setLevel(lg.DEBUG)

    # Creating Formatters
    format_ = lg.Formatter('%(asctime)s:\t %(levelname)s:\t:%(message)s')
    # Creating Handlers
    file_handler = lg.FileHandler(log_file)

    # Adding Formatters to Handlers
    file_handler.setFormatter(format_)
    # Adding Handlers to logger
    logger.addHandler(file_handler)

    stream_handler = lg.StreamHandler()
    stream_handler.setLevel(lg.INFO)
    logger.addHandler(stream_handler)
    return logger