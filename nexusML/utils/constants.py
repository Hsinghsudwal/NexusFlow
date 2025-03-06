# constants.py

import os

# Local paths
ARTIFACTS_DIR = os.path.join(os.getcwd(), 'artifacts')
MODEL_DIR = os.path.join(ARTIFACTS_DIR, 'models')

# Cloud storage configuration (if using cloud storage like S3)
USE_CLOUD_STORAGE = False
S3_BUCKET = "your-s3-bucket"
S3_MODEL_PATH = "models/"

# Other constants
LOGGING_DIR = os.path.join(os.getcwd(), 'logs')
