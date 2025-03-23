# src/config/pipeline_config.py
import os
from datetime import datetime

class PipelineConfig:
    """Configuration class for ML pipeline settings."""
    
    def __init__(self):
        # Base paths
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.MODELS_DIR = os.path.join(self.BASE_DIR, "models")
        self.LOGS_DIR = os.path.join(self.BASE_DIR, "logs")
        
        # Create directories if they don't exist
        for directory in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        # Database configuration
        self.DB_HOST = os.getenv("DB_HOST", "localhost")
        self.DB_PORT = os.getenv("DB_PORT", "5432")
        self.DB_NAME = os.getenv("DB_NAME", "churn_db")
        self.DB_USER = os.getenv("DB_USER", "postgres")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
        
        # Model configuration
        self.MODEL_TYPE = os.getenv("MODEL_TYPE", "random_forest")
        self.TRAIN_TEST_SPLIT = float(os.getenv("TRAIN_TEST_SPLIT", "0.2"))
        self.RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
        
        # Monitoring configuration
        self.MONITORING_INTERVAL = int(os.getenv("MONITORING_INTERVAL", "3600"))  # in seconds
        self.DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))
        
        # Deployment configuration
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        
        # AWS LocalStack configuration
        self.AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
        self.S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "churn-models")
        
        # Model versioning
        self.MODEL_VERSION = datetime.now().strftime("%Y%m%d%H%M%S")
        self.PROD_MODEL_PATH = os.path.join(self.MODELS_DIR, "production_model.pkl")
        
        # Retraining configuration
        self.RETRAINING_SCHEDULE = os.getenv("RETRAINING_SCHEDULE", "weekly")  # daily, weekly, monthly
        self.PERFORMANCE_THRESHOLD = float(os.getenv("PERFORMANCE_THRESHOLD", "0.01"))  # min improvement required
