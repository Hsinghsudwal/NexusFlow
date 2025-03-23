import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

# MLflow settings
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # Changed to use SQLite for simplicity
EXPERIMENT_NAME = "default_experiment"

# Model serving
MODEL_SERVING_HOST = "0.0.0.0"
MODEL_SERVING_PORT = 5000  # Changed from 8000 to 5000 to match workflow requirements

# Monitoring
PROMETHEUS_PORT = 9090

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)