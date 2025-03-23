import pandas as pd
from prefect import task
import logging

logger = logging.getLogger(__name__)

class FeatureEngineering:
    @task
    def process_features(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Process features according to configuration."""
        logger.info("Starting feature engineering")

        try:
            processed_data = data.copy()
            categorical_features = config['features']['categorical_features']
            if categorical_features:
                processed_data = pd.get_dummies(processed_data, columns=categorical_features)

            logger.info(f"Feature engineering complete. Output shape: {processed_data.shape}")
            return processed_data

        except Exception as e:
            logger.error(f"Error during feature engineering: {e}")
            raise