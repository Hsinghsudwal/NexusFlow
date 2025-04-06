# rollback_manager.py
from core.artifact_manager import ArtifactManager
from core.metadata_manager import MetadataManager

class RollbackManager:
    def __init__(self, config):
        self.artifacts = ArtifactManager(config)
        self.metadata = MetadataManager(config)

    def list_model_versions(self):
        # For simplicity, load metadata and return saved versions
        pass  # You can read from SQLite or DynamoDB and filter

    def rollback(self, model_key):
        model = self.artifacts.load(model_key)
        print(f"Rolling back to model: {model_key}")
        # Deploy the model to inference environment
        return model
