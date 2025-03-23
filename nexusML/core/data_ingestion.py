import pandas as pd
from feast import FeatureStore
import dvc.api

class DataIngestion:
    def __init__(self, config):
        self.config = config
        self.feature_store = FeatureStore(config['feature_store']['repo_path'])

    def load_data(self):
        # Load data using DVC
        with dvc.api.open(self.config['data']['path'], repo=self.config['dvc']['repo_url']) as fd:
            df = pd.read_csv(fd)
        
        # Enrich with feature store
        entity_df = df[['customer_id', 'timestamp']]  # Example entity columns
        return self.feature_store.get_historical_features(
            entity_df=entity_df,
            features=self.config['feature_store']['feature_views']
        ).to_df()