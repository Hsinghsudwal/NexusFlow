from core.artifact_store import ArtifactStore

class DataLoader:
    def __init__(self, config):
        self.artifact_store = ArtifactStore(config)
        
    def load_data(self):
        data = self.artifact_store.load_artifact("raw_data", "dataset.csv")
        return data