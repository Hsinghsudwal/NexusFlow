import mlflow
from mlflow.tracking import MlflowClient

class ModelRegistry:
    def __init__(self, config):
        self.client = MlflowClient()
        self.config = config
        
    def register_model(self, run_id, model_name):
        model_uri = f"runs:/{run_id}/model"
        return mlflow.register_model(model_uri, model_name)
    
    def promote_model(self, model_name, version, stage="Production"):
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
    
    def get_production_model(self, model_name):
        return self.client.get_latest_versions(model_name, stages=["Production"])[0]

class VersionedModel:
    def __init__(self, model, metadata):
        self.model = model
        self.metadata = {
            **metadata,
            "version": self._generate_version()
        }
        
    def _generate_version(self):
        return f"{datetime.now().strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:6]}"



