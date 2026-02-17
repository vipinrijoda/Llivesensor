import logging
import sys
import os
from datetime import datetime 

LOG_FILE_NAME = f"{datetime.now().strftime('%m%d%Y_%H%M%S')}.log"

# Create logs directory only (not with filename)
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Create full log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(levelname)s- %(message)s",
    level=logging.INFO,
)