from tecton import FeatureService

class TectonFeatureStore:
    def __init__(self, config):
        self.workspace = config.get("tecton.workspace")
        self.feature_service = FeatureService.get(
            config.get("tecton.feature_service"),
            workspace=self.workspace
        )
        
    def get_online_features(self, entity_keys):
        return self.feature_service.get_online_features(
            join_keys=entity_keys
        ).to_dict()