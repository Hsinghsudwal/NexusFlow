from feast import FeatureStore
import pandas as pd

class FeatureStoreManager:
    def __init__(self, config):
        self.store = FeatureStore(repo_path=config['feature_store_path'])

    def get_historical_features(self, entity_df: pd.DataFrame):
        return self.store.get_historical_features(
            entity_df=entity_df,
            features=self.store.get_feature_service("model_features")
        ).to_df()

    def write_online_features(self, feature_data: pd.DataFrame):
        self.store.write_to_online_store(
            feature_data,
            allow_registry_cache=False
        )

    def materialize_features(self, start_date, end_date):
        self.store.materialize(
            start_date=start_date,
            end_date=end_date
        )