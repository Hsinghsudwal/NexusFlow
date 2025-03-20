from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

@dataclass
class PipelineMetadata:
    run_id: str
    execution_date: datetime
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    git_commit: str

class MetadataTracker:
    def __init__(self, config):
        self.config = config
        self.metadata_store = None
        
    def initialize(self):
        if self.config.get("metadata_store.type") == "mlflow":
            from integrations.mlflow.mlflow_tracking import MLflowTracking
            self.metadata_store = MLflowTracking(self.config)
            
    def capture_metadata(self, metadata: PipelineMetadata):
        if self.metadata_store:
            self.metadata_store.start_run(run_name=metadata.run_id)
            self.metadata_store.log_params(metadata.parameters)
            self.metadata_store.log_metrics(metadata.metrics)
            for artifact_name, artifact_path in metadata.artifacts.items():
                self.metadata_store.log_artifact(artifact_path)
            self.metadata_store.end_run()