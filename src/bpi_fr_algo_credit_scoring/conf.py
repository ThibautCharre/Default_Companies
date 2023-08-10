import os
import logging

from datetime import datetime

ROOT = os.getcwd()
DATA_ROOT = f"{ROOT}/data"

LOG_ROOT = f"{ROOT}/logs"
LOG_FILENAME = datetime.now().strftime(f"{LOG_ROOT}/Project_log_%H_%M_%S_%d_%m_%Y.log")

log_config = logging.basicConfig(
    filename=LOG_FILENAME, encoding="UTF-8", level=logging.DEBUG
)
