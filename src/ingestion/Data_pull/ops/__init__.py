# Data_pull/ops/__init__.py
from .unzip_and_discover import unzip_and_discover
from .copy_and_insert_to_production import copy_and_insert_to_production
from .post_ingest_maintenance import post_ingest_maintenance

# Legacy: keep backward compatibility
copy_cast_to_staging = copy_and_insert_to_production

__all__ = [
    "unzip_and_discover",
    "copy_and_insert_to_production",
    "copy_cast_to_staging",  # alias for backward compatibility
    "post_ingest_maintenance",
]
