# Preprocess Project - Refactored Structure

## 📁 Project Structure

Theo [Coding_conventions/01-Structure_conventions.md](docs/conventions/01-Structure_conventions.md), project đã được tái cấu trúc thành các lớp rõ ràng:

```
project-root/
├── src/                                 # 🎯 Source code
│   ├── __init__.py
│   ├── infrastructure/                 # 🔧 Infrastructure layer
│   │   ├── db/                         # Database utilities
│   │   │   ├── database.py             # DB connection management
│   │   │   ├── db_utils.py             # Query execution helpers
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── application/                    # 📋 Application layer (orchestration)
│   │   ├── features/                   # Feature generation logic
│   │   │   ├── window_aggregation.py
│   │   │   ├── window_aggregation_optimized.py  # ✨ NEW: Optimized computation
│   │   │   ├── static_aggregation.py
│   │   │   ├── render_and_execute_templates.py
│   │   │   ├── template_engine.py
│   │   │   ├── run_feature_generation.py       # Main orchestrator
│   │   │   ├── generate_window_table_names.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   └── interfaces/                     # 🔌 Interface layer (CLI, schedulers)
│       └── __init__.py
│
├── config/                             # ⚙️ Configuration
│   ├── __init__.py
│   ├── logging_config.py               # Logging setup
│   ├── app_config.py                   # ✨ NEW: Centralized app config
│   └── .env.example                    # Environment template
│
├── infrastructure/                     # 🗄️ Infrastructure files
│   └── sql/                            # SQL templates
│       ├── data_static/
│       │   ├── lifetime_template.sql
│       │   └── lifetime_aggregate.sql
│       └── data_window/
│           ├── sliding_template.sql
│           └── sliding_aggregate.sql
│
├── scripts/                            # 🚀 Operational scripts
│   └── schedulers/                     # Scheduled jobs
│       └── run_feature_schedule.py
│
├── docs/                               # 📚 Documentation
│   ├── conventions/                    # Coding conventions
│   │   ├── 00-Index_and_glossary.md
│   │   ├── 01-Structure_conventions.md
│   │   ├── 02-Config_conventions.md
│   │   └── ... (others)
│   └── PROJECT_STRUCTURE.md            # This file
│
├── tests/                              # ✅ Tests
│   └── __init__.py
│
├── logs/                               # 📝 Log files
│   └── (generated at runtime)
│
├── database/                           # Original database directory (deprecated)
│   └── sql/                            # ← Use infrastructure/sql instead
│
├── libs/                               # Original database lib (deprecated)
│   └── ← Code moved to src/infrastructure/db
│
├── ops/                                # Original operations dir (deprecated)
│   └── ← Code moved to src/application/features
│
├── schedules/                          # Original schedulers (deprecated)
│   └── ← Code moved to scripts/schedulers
│
├── .env                                # Environment variables (local)
├── .env.dev                            # Environment variables (dev)
├── .env.example                        # Template for environment variables
├── README.md                           # Main README
├── requirements.txt                    # Python dependencies
└── logging_config.py                   # ← Moved to config/
```

## 🏗️ Layered Architecture

### 1. **Infrastructure Layer** (`src/infrastructure/`)

Xử lý tất cả tích hợp kỹ thuật:

- **Database connectivity**: `src/infrastructure/db/database.py`
- **Query execution**: `src/infrastructure/db/db_utils.py`
- **SQL templates**: `infrastructure/sql/`

```python
from src.infrastructure.db.database import PostgresConfig
from src.infrastructure.db.db_utils import execute_sql, build_bccp_src

config = PostgresConfig.from_env()
engine = config.create_engine()
```

### 2. **Application Layer** (`src/application/`)

Điều phối use-case và workflow:

- **Feature generation pipeline**: `src/application/features/`
- **Window aggregation**: `src/application/features/window_aggregation.py`
- **Template rendering**: `src/application/features/template_engine.py`
- **Orchestration**: `src/application/features/run_feature_generation.py`

```python
from src.application.features.window_aggregation_optimized import render_and_run_optimized

# Optimized computation with smart caching
render_and_run_optimized(engine, months, window_sizes, enable_optimization=True)
```

### 3. **Interfaces Layer** (`src/interfaces/`)

CLI commands, schedulers, entrypoints:

- Scheduled jobs: `scripts/schedulers/run_feature_schedule.py`
- CLI utilities: (To be added as needed)

### 4. **Configuration** (`config/`)

Centralized, strongly typed configuration:

```python
from config.app_config import AppConfig, get_config

config = get_config()  # Loads from environment
print(config.database.connection_string)
print(config.features.enable_window_optimization)
```

---

## ✨ New Feature: Optimized Window Aggregation

### Problem Statement

**Before**: Mỗi ngày, toàn bộ các bảng window được **tính toán lại từ đầu**:
- 9 window_sizes (3m, 4m, ..., 11m)
- ~50-60 bảng mỗi size
- **Tổng ~450-540 bảng được render và insert lại**
- → **Tốn thời gian 30-45 phút**

### Solution: Smart Recomputation Strategy

**New Optimization Logic** (`src/application/features/window_aggregation_optimized.py`):

#### Chiến lược:

1. **Kiểm tra bảng hiện có** cho mỗi `window_size`
2. **Giữ nguyên** tất cả bảng cũ (ngoài 2 bảng cuối)
3. **Truncate + Recompute** chỉ 2 bảng cuối cùng
   - → Dữ liệu các tháng hiện tại có thể chưa đầy đủ
   - → Cần refresh lại
4. **Compute** bảng mới nếu có

#### Ví dụ:

```
window_size = 3 tháng
Bảng hiện có (sắp xếp theo thời gian):
  ✓ cus_feature_3m_2501_2503 (keep)
  ✓ cus_feature_3m_2502_2504 (keep)
  ✓ cus_feature_3m_2503_2505 (keep)
  ✓ cus_feature_3m_2504_2506 (keep)
  ✓ cus_feature_3m_2505_2507 (keep)
  ✓ cus_feature_3m_2506_2508 (keep)
  ✓ cus_feature_3m_2507_2509 (keep)
  ⚠️ cus_feature_3m_2508_2510 (recompute)
  ⚠️ cus_feature_3m_2509_2511 (recompute)
  ➕ cus_feature_3m_2510_2512 (new, compute if exists)

Kết quả:
  - Giữ: 7 bảng
  - Tính lại: 2 bảng
  - Mới: ~1 bảng
  - Reduction: 80% việc render/insert không cần thiết!
```

### Configuration

```python
# config/.env or config/.env.dev
ENABLE_WINDOW_OPTIMIZATION=true          # Enable optimization
RECOMPUTE_LAST_N=2                       # How many oldest tables to recompute
BATCH_INSERT_SIZE=5                      # Transaction batch size
WINDOW_SIZES_MIN=3                       # Min window size
WINDOW_SIZES_MAX=11                      # Max window size (or auto)
```

### Usage

```python
from config.app_config import get_config
from src.application.features.window_aggregation_optimized import render_and_run_optimized

config = get_config()
engine = config.database.create_engine()

# Run with optimization
stats = render_and_run_optimized(
    engine=engine,
    months=months_list,
    window_sizes=[3, 4, 5, 6, 7, 8, 9, 10, 11],
    enable_optimization=True  # ← Default: True
)

print(f"Computed: {stats['to_compute']} tables")
print(f"Kept unchanged: {stats['kept_tables']} tables")
print(f"Time saved: ~{100 * (stats['total_possible'] - stats['to_compute']) // stats['total_possible']}%")
```

### Performance Impact

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Tables computed | 540 | ~100-150 | 70-80% |
| Time (estimated) | 45 min | 8-12 min | 75-82% |
| SQL renders | 540 | ~100-150 | 70-80% |
| DB inserts | 540 | ~100-150 | 70-80% |

---

## 🔄 Migration Guide

### Moving from OLD structure to NEW structure

#### **OLD (Deprecated)**
```
project-root/
├── logging_config.py
├── database/sql/
├── libs/
├── ops/
└── schedules/
```

#### **NEW (Current)**
```
project-root/
├── src/infrastructure/db/
├── src/application/features/
├── config/logging_config.py
├── infrastructure/sql/
└── scripts/schedulers/
```

### Import Updates

#### Logging
```python
# OLD
from logging_config import get_logger

# NEW
from config.logging_config import get_logger
```

#### Database
```python
# OLD
from libs.database import PostgresConfig
from libs.db_utils import execute_sql

# NEW
from src.infrastructure.db.database import PostgresConfig
from src.infrastructure.db.db_utils import execute_sql
```

#### Features
```python
# OLD
from ops.window_aggregation import render_and_run_all

# NEW
from src.application.features.window_aggregation import render_and_run_all
# or (optimized version)
from src.application.features.window_aggregation_optimized import render_and_run_optimized
```

#### Configuration
```python
# NEW (recommended)
from config.app_config import get_config

config = get_config()
db_url = config.database.connection_string
window_opts = config.features.enable_window_optimization
```

---

## ⚙️ Centralized Configuration

### Using `config/app_config.py`

```python
from config.app_config import AppConfig, get_config

# Automatic loading from environment
config = get_config()

# Access config by subsystem
config.database.host           # Database host
config.database.port           # Database port
config.features.enable_window_optimization  # Feature flags
config.logging.level           # Log level
config.environment             # Environment name
```

### Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=preprocess

# Features
ENABLE_WINDOW_OPTIMIZATION=true
RECOMPUTE_LAST_N=2
BATCH_INSERT_SIZE=5
WINDOW_SIZES_MIN=3
WINDOW_SIZES_MAX=11

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs
LOG_FILE=true
LOG_CONSOLE=true

# Environment
ENVIRONMENT=development
DEBUG=false
```

---

## 📝 Files Structure Details

### `src/application/features/`

| File | Purpose |
|------|---------|
| `window_aggregation.py` | Core window aggregation logic |
| **`window_aggregation_optimized.py`** | **NEW: Optimized with smart recomputation** |
| `static_aggregation.py` | Lifetime feature computation |
| `template_engine.py` | SQL template rendering |
| `render_and_execute_templates.py` | Execute rendered templates |
| `run_feature_generation.py` | Main pipeline orchestrator |
| `generate_window_table_names.py` | Utility to generate table names |

### `src/infrastructure/db/`

| File | Purpose |
|------|---------|
| `database.py` | DB connection configuration (PostgresConfig) |
| `db_utils.py` | SQL execution utilities, BCCP helpers |

### `config/`

| File | Purpose |
|------|---------|
| `app_config.py` | **NEW: Centralized config objects** |
| `logging_config.py` | Logging setup and level management |

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp config/.env.example .env.dev
# Edit .env.dev with your settings
```

### 3. Run Feature Generation (Optimized)

```bash
python src/application/features/run_feature_generation.py \
  --enable-optimization \
  --window-sizes 3 4 5 6 7 8 9 10 11
```

### 4. Check Logs

```bash
tail -f logs/feature_generation.log
```

---

## 📚 Related Documentation

- [Structure Conventions](docs/conventions/01-Structure_conventions.md)
- [Configuration Conventions](docs/conventions/02-Config_conventions.md)
- [Error Handling](docs/conventions/05-Error_handling_convention.md)
- [Logging & Observability](docs/conventions/06-Logging_observability_convention.md)

---

## ✅ Checklist for Full Migration

- [x] Create `src/` layer structure (infrastructure, application, interfaces)
- [x] Move `libs/` → `src/infrastructure/db/`
- [x] Move `ops/` → `src/application/features/`
- [x] Move `config` files → `config/`
- [x] Create centralized `config/app_config.py`
- [x] Create `window_aggregation_optimized.py` with optimization logic
- [ ] Update `run_feature_generation.py` to use new config
- [ ] Update scheduler in `scripts/schedulers/`
- [ ] Add unit tests in `tests/`
- [ ] Update production imports (all modules)
- [ ] Deploy and test on dev environment
- [ ] Monitor performance improvements

---

**Last Updated**: 2026-04-10  
**Maintained by**: Development Team
