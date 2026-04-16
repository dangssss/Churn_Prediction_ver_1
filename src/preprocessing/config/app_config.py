"""
Application Configuration

Centralized configuration management for the Preprocess pipeline.
Follows the configuration conventions from Coding_conventions/02-Config_conventions.md

This module provides:
- Strongly typed config objects for each subsystem
- Centralized loading from environment or settings
- Validation before use
- Safe handling of multiple environments
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from .env_loader import load_project_env_files
from .env_loader import parse_bool
from .env_loader import parse_int
from .env_loader import require_env

load_project_env_files()


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    
    host: str
    port: int
    user: str
    password: str
    dbname: str
    driver: str
    
    @property
    def connection_string(self) -> str:
        """Build SQLAlchemy connection string."""
        return f"postgresql+{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Load database config from environment variables."""
        return cls(
            host=require_env('DB_HOST'),
            port=parse_int('DB_PORT'),
            user=require_env('DB_USER'),
            password=require_env('DB_PASSWORD'),
            dbname=require_env('DB_NAME'),
            driver=require_env('DB_DRIVER')
        )


@dataclass
class FeatureGenerationConfig:
    """Feature generation pipeline configuration."""
    
    # Window aggregation settings
    window_sizes_min: int
    window_sizes_max: Optional[int]  # None = auto-calculate based on data
    enable_window_optimization: bool
    recompute_last_n_windows: int  # How many latest windows to always recompute
    
    # Static feature settings
    enable_static_features: bool
    static_data_start_date: str
    
    # Data retention
    keep_window_history: int  # Keep at least 2 versions of each window
    
    # Performance tuning
    batch_insert_size: int  # Tables per transaction
    parallel_render: bool
    window_max_workers: int
    
    @classmethod
    def from_env(cls) -> 'FeatureGenerationConfig':
        """Load feature generation config from environment."""
        window_sizes_max_raw = require_env('WINDOW_SIZES_MAX')
        return cls(
            window_sizes_min=parse_int('WINDOW_SIZES_MIN'),
            window_sizes_max=None if window_sizes_max_raw in {'', '0', 'none', 'null'} else int(window_sizes_max_raw),
            enable_window_optimization=parse_bool('ENABLE_WINDOW_OPTIMIZATION'),
            recompute_last_n_windows=parse_int('RECOMPUTE_LAST_N'),
            enable_static_features=parse_bool('ENABLE_STATIC_FEATURES'),
            static_data_start_date=require_env('STATIC_DATA_START'),
            keep_window_history=parse_int('KEEP_WINDOW_HISTORY'),
            batch_insert_size=parse_int('BATCH_INSERT_SIZE'),
            parallel_render=parse_bool('PARALLEL_RENDER'),
            window_max_workers=parse_int('WINDOW_MAX_WORKERS')
        )


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: str
    format: str
    log_dir: str
    enable_file: bool
    enable_console: bool
    
    @property
    def log_dir_path(self) -> Path:
        """Get log directory as Path object."""
        return Path(self.log_dir)
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Load logging config from environment."""
        return cls(
            level=require_env('LOG_LEVEL'),
            format=require_env('LOG_FORMAT'),
            log_dir=require_env('LOG_DIR'),
            enable_file=parse_bool('LOG_FILE'),
            enable_console=parse_bool('LOG_CONSOLE')
        )


@dataclass
class AppConfig:
    """Root application configuration.
    
    Composes all subsystem configurations into a single config object.
    """
    
    database: DatabaseConfig
    features: FeatureGenerationConfig
    logging: LoggingConfig
    
    environment: str
    debug: bool
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load complete application config from environment."""
        return cls(
            database=DatabaseConfig.from_env(),
            features=FeatureGenerationConfig.from_env(),
            logging=LoggingConfig.from_env(),
            environment=require_env('ENVIRONMENT'),
            debug=parse_bool('DEBUG')
        )
    
    def validate(self) -> None:
        """Validate configuration values.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.database.host:
            raise ValueError("Database host not configured")
        if not self.database.dbname:
            raise ValueError("Database name not configured")
        if self.features.window_sizes_min < 1:
            raise ValueError("window_sizes_min must be >= 1")
        if self.features.window_max_workers < 1:
            raise ValueError("window_max_workers must be >= 1")


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or initialize global config instance."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
        _config.validate()
    return _config


def set_config(config: AppConfig) -> None:
    """Set global config instance (useful for testing)."""
    global _config
    _config = config
