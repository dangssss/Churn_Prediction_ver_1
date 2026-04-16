from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_required_dir(env_name: str) -> Path:
    value = os.getenv(env_name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {env_name}")

    path = Path(value)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    return path

CHURN_MODEL_DIR = get_required_dir("CHURN_MODEL_DIR")
