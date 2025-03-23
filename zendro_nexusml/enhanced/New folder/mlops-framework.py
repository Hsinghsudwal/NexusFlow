"""
MLOps Framework with Parallel Pipeline Structure
===============================================
A modular framework for machine learning operations with support for:
- Parallel pipeline execution
- Artifact management
- Configuration handling
- Pipeline visualization
"""

import os
import yaml
import json
import logging
import datetime
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import time

# === Constants Module ===
class MLOpsConstants:
    """Constants used throughout the MLOps framework."""
    
    # Execution constants
    MAX_PARALLEL_PIPELINES = 4
    EXECUTOR_TIMEOUT = 3600  # seconds
    
    # Artifact constants
    ARTIFACT_ROOT = "artifacts"
    CONFIG_ROOT = "configs"
    LOGS_ROOT = "logs"
    
    # Status constants
    STATUS_PENDING = "PENDING"
    STATUS_RUNNING = "RUNNING"
    STATUS_COMPLETED = "COMPLETED"
    STATUS_FAILED = "FAILED"
    
    # Visualization constants
    VIZ_FORMATS = ["png", "html", "json"]
    VIZ_DEFAULT_WIDTH = 1200
    VIZ_DEFAULT_HEIGHT = 800


# === Configuration Module ===
class ConfigManager:
    """Manages configurations for MLOps pipelines."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(MLOpsConstants.CONFIG_ROOT, "default.yaml")
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump({
                    "version": "1.0",
                    "environment": "development",
                    "logging": {"level": "INFO"}
                }, f)
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        
        # Save the updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)


# === Artifact Management Module ===
class Artifact:
    """Represents an MLOps artifact."""
    
    def __init__(self, name: str, artifact_type: str, pipeline_id: str):
        self.name = name
        self.type = artifact_type
        self.pipeline_id = pipeline_id
        self.created_at = datetime.datetime.now()
        self.path = self._generate_path()
        self.metadata = {}
    
    def _generate_path(self) -> str:
        """Generate filesystem path for the artifact."""
        timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(
            MLOpsConstants.ARTIFACT_ROOT,
            self.pipeline_id,
            self.type
        )
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"{self.name}_{timestamp}")
    
    def save(self, data: Any, metadata: Dict = None) -> str:
        """Save artifact data and metadata."""
        # Save metadata
        self.metadata.update(metadata or {})
        self.metadata["saved_at"] = datetime.datetime.now().isoformat()
        
        # Determine file extension based on artifact type
        if self.type == "model":
            extension = "pkl"
        elif self.type == "dataset":
            extension = "csv"
        elif self.type == "metrics":
            extension = "json"
        elif self.type == "visualization":
            extension = "png"
        else:
            extension = "dat"
        
        # Create full path with extension
        full_path = f"{self.path}.{extension}"
        
        # Save data based on type
        if self.type == "metrics" or isinstance(data, (dict, list)):
            with open(full_path, 'w') as f:
                json.dump(data, f)
        elif self.type == "log":
            with open(full_path, 'w') as f:
                f.write(data)
        else:
            # Binary data
            try:
                import pickle
                with open(full_path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                logging.error(f"Failed to save artifact {self.name}: {str(e)}")
                return None
        
        # Save metadata alongside the artifact
        meta_path = f"{self.path}.metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, default=str)
        
        return full_path
    
    @staticmethod
    def load(path: str) -> tuple:
        """Load artifact and its metadata from path."""
        # Load metadata
        meta_path = path.rsplit('.', 1)[0] + ".metadata.json"
        metadata = {}
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        # Load artifact data
        extension = path.rsplit('.', 1)[1]
        if extension in ['json', 'yml', 'yaml']:
            with open(path, 'r') as f:
                if extension == 'json':
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)
        elif extension in ['txt', 'log']:
            with open(path, 'r') as f:
                data = f.read()
        else:
            # Binary data
            try:
                import pickle
                with open(path, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                logging.error(f"Failed to load artifact at {path}: {str(e)}")
                return None, metadata
        
        return data, metadata


class ArtifactManager:
    """Manages artifacts across pipelines."""
    
    def __init__(self):
        os.makedirs(MLOpsConstants.ARTIFACT_ROOT, exist_ok=True)
        self.artifacts = {}
    
    def create_artifact(self, name: str, artifact_type: str, pipeline_id: str) -> Artifact:
        """Create a new artifact."""
        artifact = Artifact(name, artifact_type, pipeline_id)
        key = f"{pipeline_id}/{artifact_type}/{name}"
        self.artifacts[key] = artifact
        return artifact
    
    def get_artifact(self, pipeline_id: str, artifact_type: str, name: str) -> Optional[Artifact]:
        """Retrieve an artifact by identifiers."""
        key = f"{pipeline_id}/{artifact_type}/{name}"
        return self.artifacts.get(key)
    
    def list_artifacts(self, pipeline_id: str = None, artifact_type: str = None) -> List[Artifact]:
        """List artifacts, optionally filtered by pipeline_id or type."""
        result = []
        
        for key, artifact in self.artifacts.items():
            if (pipeline_id is None or artifact.pipeline_id == pipeline_id) and \
               (artifact_type is None or artifact.type == artifact_type):
                result.append(artifact)
        
        return result


# === Pipeline Module ===
class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""
    
    def __init__(self, name: str):
        self.name = name
        self.status = MLOpsConstants.STATUS_PENDING
        self.start_time = None
        self.end_time = None
        self.error = None
        self.outputs = {}
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline step."""
        pass
    
    def run(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the pipeline step with timing and status tracking."""
        self.status = MLOpsConstants.STATUS_RUNNING
        self.start_time = datetime.datetime.now()
        
        try:
            self.outputs = self.execute(inputs, context)
            self.status = MLOpsConstants.STATUS_COMPLETED
            return self.outputs
        except Exception as e:
            self.status = MLOpsConstants.STATUS_FAILED
            self.error = str(e)
            logging.error(f"Step {self.name} failed: {str(e)}")
            raise
        finally:
            self.end_time = datetime.datetime.now()


class Pipeline:
    """Represents an ML pipeline with multiple steps."""
    
    def __init__(self, name: str, config: ConfigManager = None):
        self.name = name
        self.id = f"{name}_{int(time.time())}"
        self.steps = []
        self.status = MLOpsConstants.STATUS_PENDING
        self.start_time = None
        self.end_time = None
        self.context = {}
        self.config = config or ConfigManager()
        self.artifact_manager = ArtifactManager()
    
    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self
    
    def run(self, initial_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the pipeline sequentially."""
        self.status = MLOpsConstants.STATUS_RUNNING
        self.start_time = datetime.datetime.now()
        step_outputs = initial_inputs or {}
        
        try:
            for step in self.steps:
                logging.info(f"Running step: {step.name}")
                step_outputs = step.run(step_outputs, self.context)
                
                # Create an artifact for the step output
                artifact = self.artifact_manager.create_artifact(
                    name=f"{step.name}_output",
                    artifact_type="step_output",
                    pipeline_id=self.id
                )
                
                artifact.save(
                    data=step_outputs,
                    metadata={
                        "step_name": step.name,
                        "pipeline_name": self.name,
                        "status": step.status,
                        "execution_time": (step.end_time - step.start_time).total_seconds()
                    }
                )
            
            self.status = MLOpsConstants.STATUS_COMPLETED
            return step_outputs
            
        except Exception as e:
            self.status = MLOpsConstants.STATUS_FAILED
            logging.error(f"Pipeline {self.name} failed: {str(e)}")
            
            # Create a failure artifact
            artifact = self.artifact_manager.create_artifact(
                name="pipeline_failure",
                artifact_type="error",
                pipeline_id=self.id
            )
            
            artifact.save(
                data=str(e),
                metadata={
                    "pipeline_name": self.name,
                    "status": self.status,
                    "failed_step": self.steps[-1].name if self.steps else None
                }
            )
            
            raise
        finally:
            self.end_time = datetime.datetime.now()
            
            # Create a pipeline summary artifact
            summary = {
                "pipeline_name": self.name,
                "pipeline_id": self.id,
                "status": self.status,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "execution_time": (self.end_time - self.start_time).total_seconds(),
                "steps": [
                    {
                        "name": step.name,
                        "status": step.status,
                        "start_time": step.start_time.isoformat() if step.start_time else None,
                        "end_time": step.end_time.isoformat() if step.end_time else None,
                        "execution_time": (step.end_time - step.start_time).total_seconds() if step.start_time and step.end_time else None
                    } for step in self.steps
                ]
            }
            
            artifact = self.artifact_manager.create_artifact(
                name="pipeline_summary",
                artifact_type="summary",
                pipeline_id=self.id
            )
            
            artifact.save(data=summary, metadata={})


# === Parallel Execution Module ===
class PipelineExecutor:
    """Executes multiple pipelines in parallel."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or MLOpsConstants.MAX_PARALLEL_PIPELINES
        self.pipelines = {}
    
    def add_pipeline(self, pipeline: Pipeline) -> None:
        """Add a pipeline to the executor."""
        self.pipelines[pipeline.id] = pipeline
    
    def execute_all(self, inputs_map: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute all registered pipelines in parallel."""
        inputs_map = inputs_map or {}
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all pipelines
            future_to_pipeline = {
                executor.submit(
                    pipeline.run, 
                    inputs_map.get(pipeline_id, {})
                ): pipeline_id
                for pipeline_id, pipeline in self.pipelines.items()
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_pipeline):
                pipeline_id = future_to_pipeline[future]
                try:
                    results[pipeline_id] = future.result()
                except Exception as e:
                    results[pipeline_id] = {
                        "status": MLOpsConstants.STATUS_FAILED,
                        "error": str(e)
                    }
        
        return results


# === Visualization Module ===
class PipelineVisualizer:
    """Visualizes ML pipelines and their execution."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or "visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _create_pipeline_graph(self, pipeline: Pipeline) -> Dict:
        """Create a graph representation of the pipeline."""
        nodes = [{"id": "start", "label": "Start", "type": "control"}]
        edges = []
        
        for i, step in enumerate(pipeline.steps):
            step_id = f"step_{i}"
            nodes.append({
                "id": step_id,
                "label": step.name,
                "type": "step",
                "status": step.status
            })
            
            # Connect to previous node
            prev_id = "start" if i == 0 else f"step_{i-1}"
            edges.append({
                "from": prev_id,
                "to": step_id,
                "label": ""
            })
            
            # If last step, connect to end
            if i == len(pipeline.steps) - 1:
                nodes.append({"id": "end", "label": "End", "type": "control"})
                edges.append({
                    "from": step_id,
                    "to": "end",
                    "label": ""
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "pipeline_name": pipeline.name,
                "pipeline_id": pipeline.id,
                "status": pipeline.status,
                "execution_time": (pipeline.end_time - pipeline.start_time).total_seconds() if pipeline.start_time and pipeline.end_time else None
            }
        }
    
    def visualize_pipeline(self, pipeline: Pipeline, format: str = "json") -> str:
        """Visualize a pipeline and save to file."""
        graph = self._create_pipeline_graph(pipeline)
        filename = f"{pipeline.id}_visualization.{format}"
        output_path = os.path.join(self.output_dir, filename)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(graph, f, indent=2)
        elif format == "html":
            # Simple HTML visualization
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Pipeline: {pipeline.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        .pipeline {{ margin-bottom: 30px; }}
        .step {{ padding: 10px; margin: 5px; border: 1px solid #ccc; border-radius: 5px; }}
        .status-COMPLETED {{ background-color: #d4edda; }}
        .status-RUNNING {{ background-color: #fff3cd; }}
        .status-FAILED {{ background-color: #f8d7da; }}
        .status-PENDING {{ background-color: #e9ecef; }}
        .metadata {{ margin-top: 20px; border-top: 1px solid #eee; padding-top: 10px; }}
        .connector {{ height: 20px; border-left: 2px dashed #999; margin-left: 50%; }}
    </style>
</head>
<body>
    <h1>Pipeline: {pipeline.name}</h1>
    <div class="pipeline">
        <div class="metadata">
            <p><strong>ID:</strong> {pipeline.id}</p>
            <p><strong>Status:</strong> {pipeline.status}</p>
            <p><strong>Execution Time:</strong> {(pipeline.end_time - pipeline.start_time).total_seconds() if pipeline.start_time and pipeline.end_time else 'N/A'} seconds</p>
        </div>
        <div class="steps">
"""
            
            for step in pipeline.steps:
                exec_time = (step.end_time - step.start_time).total_seconds() if step.start_time and step.end_time else 'N/A'
                html_content += f"""
            <div class="connector"></div>
            <div class="step status-{step.status}">
                <h3>{step.name}</h3>
                <p><strong>Status:</strong> {step.status}</p>
                <p><strong>Execution Time:</strong> {exec_time} seconds</p>
                {f'<p><strong>Error:</strong> {step.error}</p>' if step.error else ''}
            </div>
"""
            
            html_content += """
        </div>
    </div>
</body>
</html>
"""
            with open(output_path, 'w') as f:
                f.write(html_content)
        else:
            raise ValueError(f"Unsupported visualization format: {format}")
        
        return output_path
    
    def visualize_pipeline_executor(self, executor: PipelineExecutor, format: str = "json") -> str:
        """Visualize multiple pipelines in an executor."""
        pipelines_data = []
        
        for pipeline_id, pipeline in executor.pipelines.items():
            pipeline_graph = self._create_pipeline_graph(pipeline)
            pipelines_data.append(pipeline_graph)
        
        filename = f"pipeline_executor_{int(time.time())}.{format}"
        output_path = os.path.join(self.output_dir, filename)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump({
                    "pipelines": pipelines_data,
                    "metadata": {
                        "total_pipelines": len(pipelines_data),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                }, f, indent=2)
        elif format == "html":
            # Simple HTML visualization for multiple pipelines
            html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Executor Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .pipeline { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .pipeline-header { display: flex; justify-content: space-between; margin-bottom: 15px; }
        .step { padding: 10px; margin: 5px; border: 1px solid #ccc; border-radius: 5px; }
        .status-COMPLETED { background-color: #d4edda; }
        .status-RUNNING { background-color: #fff3cd; }
        .status-FAILED { background-color: #f8d7da; }
        .status-PENDING { background-color: #e9ecef; }
        .metadata { margin-top: 20px; border-top: 1px solid #eee; padding-top: 10px; }
        .connector { height: 20px; border-left: 2px dashed #999; margin-left: 50%; }
    </style>
</head>
<body>
    <h1>Pipeline Executor Visualization</h1>
    <p>Total Pipelines: """ + str(len(pipelines_data)) + """</p>
"""
            
            for pipeline_id, pipeline in executor.pipelines.items():
                exec_time = (pipeline.end_time - pipeline.start_time).total_seconds() if pipeline.start_time and pipeline.end_time else 'N/A'
                html_content += f"""
    <div class="pipeline">
        <div class="pipeline-header">
            <h2>{pipeline.name}</h2>
            <span class="status-{pipeline.status}">{pipeline.status}</span>
        </div>
        <div class="metadata">
            <p><strong>ID:</strong> {pipeline.id}</p>
            <p><strong>Execution Time:</strong> {exec_time} seconds</p>
        </div>
        <div class="steps">
"""
                
                for step in pipeline.steps:
                    step_exec_time = (step.end_time - step.start_time).total_seconds() if step.start_time and step.end_time else 'N/A'
                    html_content += f"""
            <div class="connector"></div>
            <div class="step status-{step.status}">
                <h3>{step.name}</h3>
                <p><strong>Status:</strong> {step.status}</p>
                <p><strong>Execution Time:</strong> {step_exec_time} seconds</p>
                {f'<p><strong>Error:</strong> {step.error}</p>' if step.error else ''}
            </div>
"""
                
                html_content += """
        </div>
    </div>
"""
            
            html_content += """
</body>
</html>
"""
            with open(output_path, 'w') as f:
                f.write(html_content)
        else:
            raise ValueError(f"Unsupported visualization format: {format}")
        
        return output_path


# === Example Implementation ===
class DataLoadingStep(PipelineStep):
    """Example step for loading data."""
    
    def __init__(self, data_path: str):
        super().__init__("DataLoading")
        self.data_path = data_path
    
    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        # Simulate data loading
        logging.info(f"Loading data from {self.data_path}")
        time.sleep(1)  # Simulating work
        
        # Return simulated data
        return {
            "data": {"features": [1, 2, 3, 4, 5], "labels": [0, 1, 0, 1, 1]},
            "metadata": {"rows": 5, "columns": 2}
        }


class DataPreprocessingStep(PipelineStep):
    """Example step for preprocessing data."""
    
    def __init__(self):
        super().__init__("DataPreprocessing")
    
    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        logging.info("Preprocessing data")
        data = inputs.get("data", {})
        time.sleep(1.5)  # Simulating work
        
        # Return processed data
        return {
            "processed_data": {"features": data.get("features", []), "labels": data.get("labels", [])},
            "metadata": {"preprocessed": True}
        }


class ModelTrainingStep(PipelineStep):
    """Example step for training a model."""
    
    def __init__(self, model_type: str = "random_forest"):
        super().__init__("ModelTraining")
        self.model_type = model_type
    
    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        logging.info(f"Training {self.model_type} model")
        processed_data = inputs.get("processed_data", {})
        time.sleep(2)  # Simulating work
        
        # Return trained model
        return {
            "model": {"type": self.model_type, "trained": True},
            "metrics": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88}
        }


class ModelEvaluationStep(PipelineStep):
    """Example step for evaluating a model."""
    
    def __init__(self):
        super().__init__("ModelEvaluation")
    
    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        logging.info("Evaluating model")
        model = inputs.get("model", {})
        time.sleep(1)  # Simulating work
        
        # Return evaluation metrics
        return {
            "evaluation": {
                "metrics": {"test_accuracy": 0.83, "test_precision": 0.80, "test_recall": 0.85},
                "confusion_matrix": [[40, 10], [8, 42]]
            }
        }


class ModelDeploymentStep(PipelineStep):
    """Example step for deploying a model."""
    
    def __init__(self, deployment_target: str = "production"):
        super().__init__("ModelDeployment")
        self.deployment_target = deployment_target
    
    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        logging.info(f"Deploying model to {self.deployment_target}")
        model = inputs.get("model", {})
        evaluation = inputs.get("evaluation", {})
        time.sleep(1.5)  # Simulating work
        
        # Return deployment info
        return {
            "deployment": {
                "status": "success",
                "target": self.deployment_target,
                "timestamp": datetime.datetime.now().isoformat(),
                "model_version": "1.0"
            }
        }


def create_example_pipeline(name: str, data_path: str) -> Pipeline:
    """Create an example pipeline with standard ML steps."""
    pipeline = Pipeline(name)
    
    pipeline.add_step(DataLoadingStep(data_path))
    pipeline.add_step(DataPreprocessingStep())
    pipeline.add_step(ModelTrainingStep())
    pipeline.add_step(ModelEvaluationStep())
    pipeline.add_step(ModelDeploymentStep())
    
    return pipeline


def run_example():
    """Run a complete example with multiple parallel pipelines."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create multiple pipelines
    training_pipeline = create_example_pipeline("training_pipeline", "data/training.csv")
    validation_pipeline = create_example_pipeline("validation_pipeline", "data/validation.csv")
    experiment_pipeline = create_example_pipeline("experiment_pipeline", "data/experiment.csv")
    
    # Create executor and add pipelines
    executor = PipelineExecutor()
    executor.add_pipeline(training_pipeline)
    executor.add_pipeline(validation_pipeline)
    executor.add_pipeline(experiment_pipeline)
    
    # Execute pipelines in parallel
    results = executor.execute_all()
    
    # Visualize pipelines
    visualizer = PipelineVisualizer()
    json_path = visualizer.visualize_pipeline_executor(executor, format="json")
    html_path = visualizer.visualize_pipeline_executor(executor, format="html")
    
    logging.info(f"Pipeline visualization saved to: {json_path} and {html_path}")
    
    # Print summary
    print("\n=== Pipeline Execution Summary ===")
    for pipeline_id, result in results.items():
        pipeline = executor.pipelines[pipeline_id]
        print(f"Pipeline: {pipeline.name} ({pipeline_id})")
        print(f"Status: {pipeline.status}")
        print(f"Execution Time: {(pipeline.end_time - pipeline.start_time).total_seconds():.2f} seconds")
        print("---")
    
    return results


if __name__ == "__main__":
    run_example()
