# core/feature_store.py
from feast import FeatureStore

class FeatureManager:
    def __init__(self, config):
        self.store = FeatureStore(repo_path=config['feature_store']['repo_path'])

    def get_features(self, entity_rows):
        return self.store.get_online_features(
            features=config['feature_store']['features'],
            entity_rows=entity_rows
        ).to_df()

    def ingest_features(self, dataframe):
        self.store.ingest(config['feature_store']['feature_view'], dataframe)
