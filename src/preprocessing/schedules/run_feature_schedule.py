from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess
import sys
import os
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
SCRIPT = BASE / 'src' / 'operations' / 'run' / 'run_feature_generation.py'


def job():
    print('[SCHEDULE] Running feature generation script (noon)')
    cmd = [sys.executable, str(SCRIPT), '--start', '2025-01-01']
    # Allow DATABASE_URL from env; subprocess will inherit env
    proc = subprocess.run(cmd, env=os.environ)
    print(f'[SCHEDULE] Feature generation exited with returncode={proc.returncode}')


if __name__ == '__main__':
    scheduler = BlockingScheduler(timezone='Asia/Ho_Chi_Minh')
    # Run each day at 12:00 (noon)
    scheduler.add_job(job, 'cron', hour=12, minute=0)
    print('[INFO] Feature generation schedule started. Runs daily at 12:00 (Asia/Ho_Chi_Minh)')
    scheduler.start()
