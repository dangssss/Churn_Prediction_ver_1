# Data_pull/schedules/daily_zip_scan_schedule.py

import logging
from apscheduler.schedulers.blocking import BlockingScheduler

from Data_pull.sensors.incoming_zip_sensor import run_once_scan
from Data_pull.logging_config import get_logger

logger = get_logger(__name__)


def job():
    logger.info("Running scheduled daily scan")
    run_once_scan()
    logger.info("Finished scheduled daily scan")


def main():
    """Main function để start scheduler - dùng cho entrypoint.py"""
    # Timezone Việt Nam (UTC+7)
    scheduler = BlockingScheduler(timezone="Asia/Ho_Chi_Minh")

    # Mỗi ngày lúc 09:00 sáng
    scheduler.add_job(job, "cron", hour=9, minute=0)

    logger.info("Scheduler started. Job will run every day at 09:00 (Asia/Ho_Chi_Minh timezone)")
    scheduler.start()


if __name__ == "__main__":
    main()
