import mlflow
from typing import Dict, Any
from core.config import Config

class MLflowTracking:
    def __init__(self, config: Config):
        mlflow_config = config.get("mlflow", {})
        self.tracking_uri = mlflow_config.get("tracking_uri", "http://localhost:5000")
        self.experiment_name = mlflow_config.get("experiment_name", "default")
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: str = None, tags: Dict[str, Any] = None):
        mlflow.start_run(run_name=run_name, tags=tags)
        
    def end_run(self):
        mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]):
        mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def log_model(self, model, artifact_path: str):
        mlflow.sklearn.log_model(model, artifact_path)