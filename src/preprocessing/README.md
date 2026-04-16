Preprocess Pipeline - Feature Generation System

## 📋 Overview / Tổng quan

**EN**: This project implements an automated feature generation pipeline for machine learning, focusing on:
- **Lifetime features**: Static customer attributes computed once
- **Window features**: Sliding-window time-series aggregations (3-month to 12-month windows)
- **Performance optimization**: Intelligent recomputation to reduce daily processing time by ~75%

**VI**: Dự án này cài đặt một pipeline tự động tạo features cho machine learning, tập trung vào:
- **Lifetime features**: Các thuộc tính static của khách hàng được tính một lần
- **Window features**: Các phép tổng hợp theo sliding-window 
- **Tối ưu hiệu năng**: Cơ chế recomputation thông minh giảm 75% thời gian xử lý hàng ngày

---

## 🏗️ Project Structure / Cấu trúc Dự án

**Updated**: This project now follows [layered architecture conventions](docs/conventions/01-Structure_conventions.md).

```
Preprocess/
├── src/                              # 🎯 Source code (production)
│   ├── infrastructure/               # 🔧 Infrastructure layer
│   │   └── db/                      # Database utilities
│   ├── application/                  # 📋 Application layer  
│   │   └── features/                # Feature generation
│   │       ├── window_aggregation.py
│   │       ├── window_aggregation_optimized.py  # ✨ NEW
│   │       ├── static_aggregation.py
│   │       └── run_feature_generation.py
│   └── interfaces/                   # 🔌 Interface layer (CLI, schedulers)
│
├── config/                           # ⚙️ Configuration
│   ├── app_config.py                 # ✨ NEW: Centralized config
│   └── logging_config.py
│
├── docs/                             # 📚 Documentation
│   ├── conventions/                  # Coding conventions
│   ├── PROJECT_STRUCTURE.md          # ✨ NEW: Full architecture
│   └── WINDOW_AGGREGATION_OPTIMIZATION.md  # ✨ NEW: Optimization guide
│
├── infrastructure/                   # 🗄️ Infrastructure files
│   └── sql/                         # SQL templates
│
├── scripts/                          # 🚀 Operational scripts
│   └── schedulers/
│
├── tests/                            # ✅ Tests
├── logs/                             # 📝 Log files
├── .env & .env.dev                   # Environment configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

**See also**: [Full Project Structure Documentation](docs/PROJECT_STRUCTURE.md)

---

## 🔄 Pipeline Overview / Quy Trình Tổng Quan

### 1) Data Ingestion (09:00 daily)

**EN**: External data pull → stored in `public` schema
**VI**: Kéo dữ liệu từ nguồn ngoài → lưu vào schema `public`

- `public.cas_customer` — Monthly customer summary (tháng)
- `public.cas_info` — Contract info snapshot
- `public.cms_complaint` — Complaint records
- `public.bccp_orderitem_YYMM` — Order details by month

### 2) Feature Generation (12:00 daily)

**EN**: Generate/refresh features in two schemas
**VI**: Tạo/refresh features trong 2 schema

#### **Data Static Schema**
- `data_static.cus_lifetime` — Lifetime features (computed once from 2025-01-01)

#### **Data Window Schema**  
- `data_window.cus_feature_{W}m_{YYMM}_{YYMM}` — Sliding windows
  - W: window size (3-11 months)
  - YYMM_YYMM: start and end dates

**Example**:
```
data_window.cus_feature_3m_2501_2503   (Jan-Mar 2025)
data_window.cus_feature_3m_2502_2504   (Feb-Apr 2025)
data_window.cus_feature_3m_2503_2505   (Mar-May 2025)
... and so on
```

---

## ⚡ New Feature: Window Aggregation Optimization

[🔗 See Detailed Optimization Guide](docs/WINDOW_AGGREGATION_OPTIMIZATION.md)


### Enable Optimization
```python
from config.app_config import get_config
from src.application.features.window_aggregation_optimized import render_and_run_optimized

config = get_config()
engine = config.database.create_engine()

stats = render_and_run_optimized(
    engine=engine,
    months=months_list,
    window_sizes=[3, 4, 5, 6, 7, 8, 9, 10, 11],
    enable_optimization=True  # ← Enable smart recomputation
)
```

---

## 1) Data Ingestion / Kéo Dữ Liệu (09:00 daily)
- Data pull: chạy tự động 09:00 hàng ngày; kết quả được lưu vào schema `public` với các bảng nguồn chính:
	- `public.cas_customer` (bảng tổng hợp theo tháng)
	- `public.cas_info` (thông tin hợp đồng / snapshot, 1 bảng duy nhất)
	- `public.cms_complaint` (bảng khiếu nại, 1 bảng duy nhất)
	- `public.bccp_orderitem_YYMM` (bảng chi tiết gửi hàng theo tháng, ví dụ `bccp_orderitem_2412`, `bccp_orderitem_2501`, ...)

2) Feature Generation / Tạo Features (12:00 daily)
- Feature generation: chạy tự động 12:00 hàng ngày; tạo/refresh hai schema feature:
	- `data_static` chứa `cus_lifetime` — lifetime/static features (chỉ xét dữ liệu từ 2025-01-01 trở đi)
	- `data_window` chứa `cus_feature_{W}m_{YYMM}_{YYMM}` — sliding-window features

2) Quy Ước và Business Rules / Business Rules (Summary)
- Window table name: `data_window.cus_feature_{W}m_{YYMM}_{YYMM}` (W = số tháng, YYMM_YYMM: bắt đầu_kết thúc).
- W được tính tự động từ `3` tới `(số tháng từ 2025-01 tới current_date) - 2` (ví dụ tháng 12/2025 → W từ 3 tới 9).
- Các bảng nguồn `public.*` là nguồn truth (cas_customer, cas_info, cms_complaint, bccp_orderitem_YYMM).

3) Main Source Code / Mã Nguồn Chính

**Orchestrator (Entry Point)**:
- [src/application/features/run_feature_generation.py](src/application/features/run_feature_generation.py) — Main pipeline entry point

**Feature Computation**:
- [src/application/features/window_aggregation.py](src/application/features/window_aggregation.py) — Core window aggregation
- [src/application/features/window_aggregation_optimized.py](src/application/features/window_aggregation_optimized.py) — **NEW: Optimized version**
- [src/application/features/static_aggregation.py](src/application/features/static_aggregation.py) — Lifetime features

**Infrastructure & Utilities**:
- [src/infrastructure/db/database.py](src/infrastructure/db/database.py) — Database configuration
- [src/infrastructure/db/db_utils.py](src/infrastructure/db/db_utils.py) — Query utilities
- [src/application/features/template_engine.py](src/application/features/template_engine.py) — SQL template rendering

**SQL Templates**:
- [infrastructure/sql/data_window/](infrastructure/sql/data_window/) — Window aggregation templates
- [infrastructure/sql/data_static/](infrastructure/sql/data_static/) — Static feature templates

**Configuration**:
- [config/app_config.py](config/app_config.py) — **NEW: Centralized application config**
- [config/logging_config.py](config/logging_config.py) — Logging configuration

---

## 🚀 Getting Started / Bắt Đầu

### 1. Install Dependencies / Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

### 2. Configure Environment / Cấu Hình Môi Trường

```bash
# Copy template
cp config/.env.example .env.dev

# Edit with your database settings
# Edit .env.dev with your specific configuration
```

**Required variables** (`.env` or `.env.dev`):
```bash
# Database
DB_HOST=your_host
DB_PORT=5432
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=preprocess

# Features (optional - defaults provided)
ENABLE_WINDOW_OPTIMIZATION=true
RECOMPUTE_LAST_N=2
BATCH_INSERT_SIZE=5

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs
```

### 3. Run Feature Generation / Chạy Tạo Features

#### **Standard Approach (with optimization)**
```bash
cd src/application/features
python run_feature_generation.py
```

#### **Programmatic Approach**
```python
from config.app_config import get_config
from src.application.features.window_aggregation_optimized import render_and_run_optimized

config = get_config()
engine = config.database.create_engine()

# Get months and window sizes
from datetime import datetime
import pandas as pd

months = pd.date_range('2025-01', periods=12, freq='MS')
window_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 11]

# Run optimized computation
stats = render_and_run_optimized(
    engine=engine,
    months=months,
    window_sizes=window_sizes,
    enable_optimization=True
)

print(f"✓ Computed {stats['to_compute']} tables in {stats['duration_seconds']:.2f}s")
```

### 4. Monitor & Verify / Giám Sát & Xác Minh

```bash
# Check logs
tail -f logs/feature_generation.log

# Verify database
# Connect to database and check:
SELECT * FROM data_window.cus_feature_3m_2501_2503;  -- Recent data
SELECT COUNT(*) FROM data_static.cus_lifetime;       -- Static features
```

---

## 📊 Configuration / Cấu Hình

All configuration is centralized in [config/app_config.py](config/app_config.py) using environment variables.

### Feature Generation Options
```bash
# Enable/disable window optimization (default: true)
ENABLE_WINDOW_OPTIMIZATION=true|false

# How many latest tables to always recompute (default: 2)
RECOMPUTE_LAST_N=2

# Batch size for inserts per transaction (default: 5)
BATCH_INSERT_SIZE=5

# Window size range
WINDOW_SIZES_MIN=3        # Minimum window size
WINDOW_SIZES_MAX=11       # Maximum window size
```

See [Full Configuration Guide](docs/PROJECT_STRUCTURE.md#-centralized-configuration) for all options.

---

## 📚 Documentation / Tài Liệu

- [Project Structure](docs/PROJECT_STRUCTURE.md) — New layered architecture
- [Window Aggregation Optimization](docs/WINDOW_AGGREGATION_OPTIMIZATION.md) — Detailed technical guide
- [Coding Conventions](docs/conventions/) — Code standards and patterns

---

## 🔗 Key Files / File Quan Trọng

| File | Purpose |  
|------|---------|
| `src/application/features/run_feature_generation.py` | Main orchestrator/entry point |
| `src/application/features/window_aggregation.py` | Core computation logic |
| `src/application/features/window_aggregation_optimized.py` | **NEW** Optimized computation |
| `config/app_config.py` | **NEW** Centralized configuration |
| `src/infrastructure/db/database.py` | Database connection management |
| `infrastructure/sql/` | SQL templates |

---

## ❓ Troubleshooting / Khắc Phục Sự Cố

### Issue: "Module not found" errors

**Solution**: Ensure you're running from the project root directory and have all dependencies installed.
```bash
pip install -r requirements.txt
cd /path/to/Preprocess
python -m src.application.features.run_feature_generation
```

### Issue: Database connection fails

**Solution**: Verify `.env` file has correct credentials:
```bash
cat .env | grep DB_
```

### Issue: Optimization disabled

**Solution**: Check feature configuration:
```bash
echo "ENABLE_WINDOW_OPTIMIZATION=$ENABLE_WINDOW_OPTIMIZATION"
# Should print: ENABLE_WINDOW_OPTIMIZATION=true
```

---

## 📈 Performance Metrics / Thống Kê Hiệu Năng

### Before Optimization
- **Daily computation**: 540+ tables
- **Processing time**: ~45 minutes
- **I/O operations**: ~540 SQL renders + 540 inserts

### After Optimization  
- **Daily computation**: ~18-20 tables (only latest + new)
- **Processing time**: ~8 minutes
- **Reduction**: **75-80% faster**, 95% fewer redundant operations

### Monitoring

Enable detailed logging to monitor performance:
```bash
LOG_LEVEL=DEBUG  # For detailed execution trace
```

Check logs for optimization statistics:
```bash
grep "Optimization summary" logs/feature_generation.log
```

---

## 🤝 Contributing / Đóng Góp

When making changes:
1. Follow [Coding Conventions](docs/conventions/)
2. Test changes locally
3. Update documentation
4. Commit with clear messages

---

## 📝 License / Giấy Phép

[Add your license here]

---

**Last Updated**: 2026-04-10  
**Maintained by**: Development Team