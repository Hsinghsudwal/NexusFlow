import pandas as pd
from pathlib import Path
from typing import Optional
import dvc.api
from src.utils import logger
from src.config import RAW_DATA_DIR

class DataIngestion:
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or RAW_DATA_DIR

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from the specified file."""
        try:
            file_path = self.data_path / filename
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save data to the specified file and version it with DVC."""
        try:
            file_path = self.data_path / filename
            df.to_csv(file_path, index=False)
            
            # Add file to DVC
            import subprocess
            subprocess.run(['dvc', 'add', str(file_path)])
            subprocess.run(['dvc', 'push'])
            
            logger.info(f"Successfully saved and versioned data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def get_versioned_data(self, filename: str, version: str) -> pd.DataFrame:
        """Retrieve specific version of data using DVC."""
        try:
            file_path = self.data_path / filename
            data = dvc.api.read(
                path=str(file_path),
                rev=version,
                mode='r'
            )
            return pd.read_csv(data)
        except Exception as e:
            logger.error(f"Error retrieving versioned data: {str(e)}")
            raise
