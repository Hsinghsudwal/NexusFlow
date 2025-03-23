import pandas as pd
from feast import FeatureStore
import dvc.api
from prefect import task
import logging

logger = logging.getLogger(__name__)

class DataIngestion:
    @task
    def data_ingestion(self, path: str, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data using DVC and enrich with feature store."""
        logger.info(f"Loading data from {path}")

        try:
            # Load data using DVC
            with dvc.api.open(path, repo=config['dvc']['remote_url']) as fd:
                data = pd.read_csv(fd)

            # Enrich with feature store
            feature_store = FeatureStore(config['feature_store']['repo_path'])
            entity_df = data[['customer_id', 'timestamp']]  # Example entity columns
            enriched_data = feature_store.get_historical_features(
                entity_df=entity_df,
                features=config['feature_store']['feature_views']
            ).to_df()

            # Split data
            train_size = config['data']['train_size']
            target_column = config['data']['target_column']
            X = enriched_data.drop(columns=[target_column])
            y = enriched_data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

            logger.info(f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
            return train_data, test_data

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise