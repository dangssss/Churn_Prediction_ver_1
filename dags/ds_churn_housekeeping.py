"""
DAG: ds_churn_housekeeping
Dọn dẹp định kỳ các runtime folders
Schedule: 03:00 hàng ngày
Các thư mục chạy trong docker nên lấy thông tin từ env.dev và env.prod
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from pendulum import datetime
import os

# Biến môi trường (SAVED_DIR, FAIL_DIR, INCOMING_DIR, ...) đã được
# docker-compose nạp từ root .env (env_file) vào container.
airflow_env = os.environ.copy()

# Housekeeping script nội dung
HOUSEKEEPING_SCRIPT = '''
#!/bin/bash
set -e

# Configuration
BUNDLE_KEEP_COUNT=10
LOG_RETENTION_DAYS=30
SAVED_RETENTION_DAYS=90
FAIL_RETENTION_DAYS=30
INCOMING_RETENTION_DAYS=7

# CÁC ĐƯỜNG DẪN LẤY TỪ BIẾN MÔI TRƯỜNG (với fallback nếu biến không tồn tại)
BUNDLE_DIR="${BUNDLE_DIR:-/churn_source/modeling/bundles}"
APP_LOGS_DIR="/logs" # Quét toàn bộ /logs để dọn cả ingest, preprocess, modeling
SAVED_DIR="${SAVED_DIR:-/churn_process/saved_data}"
FAIL_DIR="${FAIL_DIR:-/churn_process/fail_ingest}"
INCOMING_DIR="${INCOMING_DIR:-/churn_data/churn_de}"

echo "=== DS_CHURN Housekeeping - $(date) ==="

# 1. Bundle retention
echo "[1] Cleaning old bundles at: ${BUNDLE_DIR}"
if [[ -d "${BUNDLE_DIR}" ]]; then
    # Đếm số lượng folder (mỗi bundle là 1 folder)
    bundle_count=$(find "${BUNDLE_DIR}" -maxdepth 1 -type d ! -path "${BUNDLE_DIR}" | wc -l)
    
    if [[ ${bundle_count} -gt ${BUNDLE_KEEP_COUNT} ]]; then
        echo "   Found ${bundle_count} bundles. Keeping last ${BUNDLE_KEEP_COUNT}..."
        # List theo thời gian sửa đổi -> sort -> lấy các dòng thừa -> xóa
        find "${BUNDLE_DIR}" -maxdepth 1 -type d ! -path "${BUNDLE_DIR}" -printf '%T@ %p\\n' | \
            sort -n | head -n -${BUNDLE_KEEP_COUNT} | cut -d' ' -f2- | xargs -r rm -rf
        echo "   Cleaned old bundles."
    else
        echo "   Bundle count (${bundle_count}) <= limit. Skipping."
    fi
else
    echo "   Warning: Bundle dir ${BUNDLE_DIR} not found!"
fi

# 2. Log rotation (App Logs: Ingest, Preprocess, Modeling...)
echo "[2] Cleaning old logs at: ${APP_LOGS_DIR}"
if [[ -d "${APP_LOGS_DIR}" ]]; then
    find "${APP_LOGS_DIR}" -type f \\( -name "*.log" -o -name "*.log.*" \\) -mtime +${LOG_RETENTION_DAYS} -delete 2>/dev/null || true
    find "${APP_LOGS_DIR}" -type d -empty -delete 2>/dev/null || true
    echo "   Cleaned logs older than ${LOG_RETENTION_DAYS} days."
fi

# 3. Saved data retention (Data đã xử lý thành công)
echo "[3] Cleaning old saved data at: ${SAVED_DIR}" 
if [[ -d "${SAVED_DIR}" ]]; then
    find "${SAVED_DIR}" -type f -mtime +${SAVED_RETENTION_DAYS} -delete 2>/dev/null || true
    find "${SAVED_DIR}" -type d -empty -delete 2>/dev/null || true
    echo "   Cleaned saved data older than ${SAVED_RETENTION_DAYS} days."
fi

# 4. Fail data retention (Data lỗi)
echo "[4] Cleaning old fail data at: ${FAIL_DIR}"
if [[ -d "${FAIL_DIR}" ]]; then
    find "${FAIL_DIR}" -type f -mtime +${FAIL_RETENTION_DAYS} -delete 2>/dev/null || true
    find "${FAIL_DIR}" -type d -empty -delete 2>/dev/null || true
    echo "   Cleaned fail data older than ${FAIL_RETENTION_DAYS} days."
fi

# 5. Incoming data retention (File zip gốc quá cũ chưa ai xử lý?)
echo "[5] Cleaning old incoming data at: ${INCOMING_DIR}"
if [[ -d "${INCOMING_DIR}" ]]; then
    find "${INCOMING_DIR}" -type f -mtime +${INCOMING_RETENTION_DAYS} -delete 2>/dev/null || true
    echo "   Cleaned incoming files older than ${INCOMING_RETENTION_DAYS} days."
fi

echo "=== Housekeeping Complete ==="
'''

with DAG(
    dag_id="ds_churn_housekeeping",
    start_date=datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    schedule="0 3 1 */3 *",  # 03:00 vào ngày mùng 1 của mỗi 3 tháng
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0},
    tags=["ds_churn", "maintenance"],
) as dag:

    housekeeping = BashOperator(
        task_id="run_housekeeping",
        bash_command=HOUSEKEEPING_SCRIPT,
        env=airflow_env,
    )
