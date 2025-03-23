import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
from src.utils import logger
from src.config import PROCESSED_DATA_DIR

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess_data(
        self, 
        df: pd.DataFrame,
        target_column: str,
        features: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the data for model training."""
        try:
            # Basic preprocessing steps
            # 1. Handle missing values
            df = df.fillna(df.mean())
            
            # 2. Extract features and target
            X = df[features]
            y = df[target_column]
            
            # 3. Scale features
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns
            )
            
            logger.info("Data preprocessing completed successfully")
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
            
    def save_processed_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        filename_prefix: str
    ) -> None:
        """Save processed data to files."""
        try:
            X_path = PROCESSED_DATA_DIR / f"{filename_prefix}_features.csv"
            y_path = PROCESSED_DATA_DIR / f"{filename_prefix}_target.csv"
            
            X.to_csv(X_path, index=False)
            y.to_csv(y_path, index=False)
            
            logger.info(f"Processed data saved to {PROCESSED_DATA_DIR}")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
