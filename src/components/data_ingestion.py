import os
import pandas as pd
import hashlib
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, Union

from utils.logger import logger
from src.core.config_manager import ConfigManager
from src.core.artifact_manager import ArtifactManager


class DataIngestion:
    """Data ingestion component with proper versioning and caching."""

    def __init__(self, data_path: str):
        """
        Initialize the data ingestion component.
        """
        self.data_path = data_path
        logger.info(f"DataIngestion initialized with data path: {self.data_path}")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from a file into a pandas DataFrame."""
        logger.info(f"Loading data from {file_path}")
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        try:
            if file_path.endswith(".csv"):
                return pd.read_csv(file_path)
            elif file_path.endswith(".parquet"):
                return pd.read_parquet(file_path)
            elif file_path.endswith((".xls", ".xlsx")):
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def validate_data(
        self, data: pd.DataFrame, target_column: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate the loaded data for quality and consistency."""
        logger.info("Validating data...")
        if data.empty:
            return False, "Dataset is empty"
        if target_column and target_column not in data.columns:
            return False, f"Target column '{target_column}' not found"
        missing_pct = data.isnull().mean()
        high_missing_cols = missing_pct[missing_pct > 0.8].index.tolist()
        if high_missing_cols:
            logger.warning(f"Columns with >80% missing: {high_missing_cols}")
        return True, None

    def hash_dataframe(self, df: pd.DataFrame) -> str:
        """Generate a hash for a dataframe to use as a unique identifier."""
        try:
            df_json = df.to_json(orient="records")
            return hashlib.sha256(df_json.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing dataframe: {e}")
            raise

    def generate_data_metadata(self, df: pd.DataFrame, source: str) -> Dict[str, Any]:
        """Generate comprehensive metadata for a dataframe."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_hash = self.hash_dataframe(df)
        
        # Create summary statistics
        numeric_stats = df.describe().to_dict() if not df.empty else {}
        
        # Calculate missing values
        missing_values = df.isnull().sum().to_dict() if not df.empty else {}
        missing_percentages = (df.isnull().mean() * 100).to_dict() if not df.empty else {}
        
        # Get column data types
        column_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        metadata = {
            "timestamp": timestamp,
            "source": source,
            "hash": data_hash,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": column_dtypes,
            "missing_values": missing_values,
            "missing_percentages": missing_percentages,
            "numeric_stats": numeric_stats,
            "version_id": f"version_{timestamp}_{data_hash[:8]}",
        }
        
        logger.info(f"Generated data metadata with version ID: {metadata['version_id']}")
        return metadata

    def split_data(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and test sets."""
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=test_size, random_state=42)

    def data_ingestion(
        self,
        results: dict,
        config_dict: dict,
        artifact_store: ArtifactManager,
        pipeline_id: str,
    ):
        """
        Perform data ingestion with proper versioning and caching.
        """
        logger.info("======== Data Ingestion ========")

        try:
            # Create ConfigManager from dict if needed
            config = (
                config_dict
                if isinstance(config_dict, ConfigManager)
                else ConfigManager(config_dict)
            )

            # Get paths from configuration
            raw_path_dir = config.get("artifact_path", {}).get("raw_path", {})
            raw_train_artifact_name = config.get("artifact_path", {}).get(
                "train", {}
            )
            raw_test_artifact_name = config.get("artifact_path", {}).get(
                "test", {}
            )
            metadata_artifact_name = "data_metadata.json"
            target_column = config.get("base", {}).get("target_column")

            # Load data
            df = self.load_data(self.data_path)
            logger.info(f"Loaded data shape: {df.shape}")

            # Generate a data hash
            current_hash = self.hash_dataframe(df)
            version_path = os.path.join(raw_path_dir, f"version_{current_hash}")

            # Try to load existing artifacts
            cache_hit = False
            try:
                # We look for artifacts in the version-specific path
                existing_metadata = artifact_store.load(version_path, metadata_artifact_name)
                train_data = artifact_store.load(raw_path_dir, raw_train_artifact_name)
                test_data = artifact_store.load(raw_path_dir, raw_test_artifact_name)

                if all([existing_metadata, train_data is not None, test_data is not None]):
                    logger.info(f"Reusing cached artifacts for hash: {current_hash}")
                    cache_hit = True
                    
                    # Return cached results
                    result = {
                        "train_data": train_data,
                        "test_data": test_data,
                        "data_metadata": existing_metadata,
                        "data_hash": current_hash
                    }
                        
                    return result
                    
            except (FileNotFoundError, Exception) as e:
                logger.info(
                    f"No cached artifacts found: {e}. Proceeding with new ingestion."
                )

            # Validate the data
            is_valid, error = self.validate_data(df, target_column)
            if not is_valid:
                error_msg = f"Data validation failed: {error}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Split the data
            train_data, test_data = self.split_data(df)
            
            # Generate comprehensive metadata
            metadata = self.generate_data_metadata(df, self.data_path)
            metadata["data_hash"] = current_hash
            metadata["split_ratio"] = {"train": 0.8, "test": 0.2}
            metadata["train_shape"] = train_data.shape
            metadata["test_shape"] = test_data.shape

            # Save artifacts
            artifact_store.save(train_data, raw_path_dir, raw_train_artifact_name, pipeline_id)
            artifact_store.save(test_data, raw_path_dir, raw_test_artifact_name, pipeline_id)
            artifact_store.save(metadata, version_path, metadata_artifact_name, pipeline_id)

            logger.info(f"Data ingestion complete. Data hash: {current_hash}")
                
            # Return results
            return {
                "train_data": train_data,
                "test_data": test_data,
                "data_metadata": metadata,
                "data_hash": current_hash
            }
            
        except Exception as e:
            logger.error(f"Error in data ingestion: {str(e)}")
            raise
