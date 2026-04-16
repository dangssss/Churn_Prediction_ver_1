# Ingestion/run_job_now.py
import logging
import sys
from pathlib import Path

# Add the current directory to sys.path to allow imports from Data_pull
# Assuming this script is at d:\ds_churn\Ingestion\run_job_now.py
# and we need to import Data_pull which is in d:\ds_churn\Ingestion
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from Data_pull.sensors.incoming_zip_sensor import run_once_scan
from Data_pull.logging_config import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Manual trigger: run_job_now.py started.")
    try:
        run_once_scan()
        logger.info("Manual trigger: run_job_now.py finished successfully.")
    except Exception as e:
        logger.error(f"Manual trigger failed: {e}")
        raise

if __name__ == "__main__":
    # Configure basic logging to stdout if not already configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
