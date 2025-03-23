import mlflow
import pandas as pd
from typing import List, Optional
from src.utils import logger
from src.config import MLFLOW_TRACKING_URI

class ModelPredictor:
    def __init__(self, model_uri: str):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.model_uri = model_uri
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the model with error handling."""
        try:
            self.model = mlflow.pyfunc.load_model(self.model_uri)
            logger.info(f"Successfully loaded model from {self.model_uri}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Will return dummy predictions.")
            self.model = None

    def predict(self, features: pd.DataFrame) -> List[float]:
        """Make predictions using the loaded model."""
        try:
            if self.model is None:
                # Return dummy predictions if model isn't available
                logger.warning("No model available, returning dummy predictions")
                return [0.0] * len(features)

            predictions = self.model.predict(features)
            logger.info(f"Generated predictions for {len(features)} samples")
            return predictions.tolist()
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise