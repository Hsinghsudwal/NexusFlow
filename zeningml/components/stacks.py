from typing import Dict, Any
from core.config import Config
from core.artifact_store import ArtifactStore

class Stack:
    def __init__(self, config: Dict[str, Any]):
        self.config = Config(config)
        self.artifact_store = ArtifactStore(self.config)
        self.orchestrator = None
        self.metadata_store = None

    def set_orchestrator(self, orchestrator):
        self.orchestrator = orchestrator

    def set_metadata_store(self, metadata_store):
        self.metadata_store = metadata_store


class Stack:
    def __init__(self, config: Dict[str, Any]):
        self.config = Config(config)
        self.versioner = PipelineVersioner()  # New
        self.metadata_tracker = MetadataTracker(self.config)  # New
        self.metadata_tracker.initialize()
        
        # Initialize artifact store with versioning
        self.artifact_store = ArtifactStore(
            self.config, 
            self.versioner.get_version_info()["run_id"]
        )

class Stack:
    def __init__(self, config: Dict[str, Any]):
        self.config = Config(config)
        
        # Initialize all components
        self.feature_store = FeatureStoreManager(config)
        self.validator = DataValidator(config)
        self.monitor = ModelPerformanceMonitor(config)
        self.cicd = MLCICD(config)
        
        # Initialize drift detectors after model training
        self.drift_detectors = {
            'data_drift': None,
            'concept_drift': None
        }