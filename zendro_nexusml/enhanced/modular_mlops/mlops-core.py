# core/artifact_manager.py
import os
import json
import datetime
import hashlib
import shutil
from typing import Dict, Any, Optional, List, Union
import pickle

class ArtifactManager:
    def __init__(self, base_dir: str = "artifacts"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def save_artifact(self, 
                     artifact: Any, 
                     name: str, 
                     artifact_type: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save an artifact with metadata and versioning.
        
        Args:
            artifact: The artifact object to save
            name: Name of the artifact
            artifact_type: Type of artifact (e.g., 'model', 'dataset', 'metrics')
            metadata: Additional metadata to store with the artifact
            
        Returns:
            The path where the artifact was saved
        """
        # Create timestamp and version
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory for this artifact
        artifact_dir = os.path.join(self.base_dir, artifact_type, name)
        version_dir = os.path.join(artifact_dir, timestamp)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save the artifact
        artifact_path = os.path.join(version_dir, f"{name}.pkl")
        
        with open(artifact_path, 'wb') as f:
            pickle.dump(artifact, f)
            
        # Compute hash of the file for integrity verification
        file_hash = self._compute_file_hash(artifact_path)
        
        # Create and save metadata
        if metadata is None:
            metadata = {}
            
        full_metadata = {
            "name": name,
            "type": artifact_type,
            "timestamp": timestamp,
            "file_hash": file_hash,
            "path": artifact_path,
            **metadata
        }
        
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
            
        # Update the latest version pointer
        with open(os.path.join(artifact_dir, "latest_version.txt"), 'w') as f:
            f.write(timestamp)
            
        return artifact_path
    
    def load_artifact(self, 
                     name: str, 
                     artifact_type: str, 
                     version: Optional[str] = None) -> Any:
        """
        Load an artifact by name, type, and optionally version.
        
        Args:
            name: Name of the artifact
            artifact_type: Type of artifact (e.g., 'model', 'dataset', 'metrics')
            version: Specific version to load, if None loads the latest
            
        Returns:
            The loaded artifact
        """
        artifact_dir = os.path.join(self.base_dir, artifact_type, name)
        
        # Determine which version to load
        if version is None:
            # Load the latest version
            try:
                with open(os.path.join(artifact_dir, "latest_version.txt"), 'r') as f:
                    version = f.read().strip()
            except FileNotFoundError:
                raise ValueError(f"No versions found for artifact {name} of type {artifact_type}")
                
        version_dir = os.path.join(artifact_dir, version)
        artifact_path = os.path.join(version_dir, f"{name}.pkl")
        
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
            
        # Verify file integrity
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        current_hash = self._compute_file_hash(artifact_path)
        if current_hash != metadata.get("file_hash"):
            raise ValueError(f"Artifact file integrity check failed for {artifact_path}")
            
        # Load the artifact
        with open(artifact_path, 'rb') as f:
            artifact = pickle.load(f)
            
        return artifact
    
    def get_artifact_metadata(self, 
                            name: str, 
                            artifact_type: str, 
                            version: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for an artifact."""
        artifact_dir = os.path.join(self.base_dir, artifact_type, name)
        
        # Determine which version to load
        if version is None:
            # Load the latest version
            try:
                with open(os.path.join(artifact_dir, "latest_version.txt"), 'r') as f:
                    version = f.read().strip()
            except FileNotFoundError:
                raise ValueError(f"No versions found for artifact {name} of type {artifact_type}")
                
        version_dir = os.path.join(artifact_dir, version)
        metadata_path = os.path.join(version_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return metadata
    
    def list_artifact_versions(self, name: str, artifact_type: str) -> List[str]:
        """List all versions of an artifact."""
        artifact_dir = os.path.join(self.base_dir, artifact_type, name)
        
        if not os.path.exists(artifact_dir):
            return []
            
        # Get all directories in the artifact directory
        versions = [d for d in os.listdir(artifact_dir) 
                   if os.path.isdir(os.path.join(artifact_dir, d)) and 
                   d != "__pycache__"]
        
        # Sort by timestamp (which is the directory name)
        versions.sort()
        
        return versions
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute a hash of a file for integrity verification."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


# core/experiment_tracker.py
import os
import json
import datetime
import uuid
from typing import Dict, Any, Optional, List, Union


class ExperimentTracker:
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)
        
    def create_experiment(self, 
                         name: str, 
                         description: Optional[str] = None, 
                         tags: Optional[List[str]] = None) -> str:
        """
        Create a new experiment.
        
        Args:
            name: Name of the experiment
            description: Optional description
            tags: Optional tags for categorization
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        experiment_data = {
            "id": experiment_id,
            "name": name,
            "description": description or "",
            "tags": tags or [],
            "created_at": timestamp,
            "updated_at": timestamp,
            "status": "created",
            "runs": []
        }
        
        # Save experiment metadata
        experiment_path = os.path.join(self.experiments_dir, f"{experiment_id}.json")
        with open(experiment_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)
            
        return experiment_id
    
    def log_run(self, 
               experiment_id: str, 
               params: Dict[str, Any], 
               metrics: Optional[Dict[str, float]] = None,
               artifacts: Optional[Dict[str, str]] = None) -> str:
        """
        Log a run within an experiment.
        
        Args:
            experiment_id: ID of the experiment
            params: Model or pipeline parameters
            metrics: Evaluation metrics
            artifacts: Paths to artifacts produced by this run
            
        Returns:
            Run ID
        """
        experiment_path = os.path.join(self.experiments_dir, f"{experiment_id}.json")
        
        if not os.path.exists(experiment_path):
            raise ValueError(f"Experiment not found: {experiment_id}")
            
        with open(experiment_path, 'r') as f:
            experiment_data = json.load(f)
            
        # Create a new run
        run_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        run_data = {
            "id": run_id,
            "created_at": timestamp,
            "params": params,
            "metrics": metrics or {},
            "artifacts": artifacts or {},
            "status": "completed"
        }
        
        # Add the run to the experiment
        experiment_data["runs"].append(run_data)
        experiment_data["updated_at"] = timestamp
        
        # Update the experiment file
        with open(experiment_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)
            
        return run_id
    
    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment data by ID."""
        experiment_path = os.path.join(self.experiments_dir, f"{experiment_id}.json")
        
        if not os.path.exists(experiment_path):
            raise ValueError(f"Experiment not found: {experiment_id}")
            
        with open(experiment_path, 'r') as f:
            experiment_data = json.load(f)
            
        return experiment_data
    
    def list_experiments(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments, optionally filtered by tag."""
        experiments = []
        
        for filename in os.listdir(self.experiments_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.experiments_dir, filename)
                
                with open(file_path, 'r') as f:
                    experiment_data = json.load(f)
                    
                # Filter by tag if specified
                if tag is None or tag in experiment_data.get("tags", []):
                    # Add a summary version (without the runs for efficiency)
                    summary = {k: v for k, v in experiment_data.items() if k != "runs"}
                    summary["run_count"] = len(experiment_data.get("runs", []))
                    experiments.append(summary)
                    
        return experiments
    
    def get_best_run(self, experiment_id: str, metric: str, higher_is_better: bool = True) -> Dict[str, Any]:
        """Get the best run from an experiment based on a specific metric."""
        experiment_data = self.get_experiment(experiment_id)
        
        if not experiment_data.get("runs"):
            raise ValueError(f"No runs found for experiment: {experiment_id}")
            
        runs = experiment_data["runs"]
        
        # Filter runs that have the specified metric
        valid_runs = [run for run in runs if metric in run.get("metrics", {})]
        
        if not valid_runs:
            raise ValueError(f"No runs with metric '{metric}' found in experiment: {experiment_id}")
            
        # Find the best run
        if higher_is_better:
            best_run = max(valid_runs, key=lambda x: x["metrics"][metric])
        else:
            best_run = min(valid_runs, key=lambda x: x["metrics"][metric])
            
        return best_run


# core/model_registry.py
import os
import json
import datetime
import shutil
from typing import Dict, Any, Optional, List, Union


class ModelRegistry:
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self.models_dir = os.path.join(registry_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.registry_file = os.path.join(registry_dir, "registry.json")
        
        # Initialize registry file if it doesn't exist
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, 'w') as f:
                json.dump({"models": {}}, f, indent=2)
                
    def register_model(self, 
                      name: str, 
                      version: str, 
                      model_path: str, 
                      metadata: Optional[Dict[str, Any]] = None,
                      description: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Register a model in the registry.
        
        Args:
            name: Name of the model
            version: Version string
            model_path: Path to the model file or directory
            metadata: Additional metadata (e.g., metrics, parameters)
            description: Optional description
            tags: Optional tags
            
        Returns:
            Model registration info
        """
        with open(self.registry_file, 'r') as f:
            registry = json.load(f)
            
        # Create model entry if it doesn't exist
        if name not in registry["models"]:
            registry["models"][name] = {"versions": {}}
            
        # Check if version already exists
        if version in registry["models"][name]["versions"]:
            raise ValueError(f"Version {version} already exists for model {name}")
            
        # Create directory for this model version
        model_dir = os.path.join(self.models_dir, name, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy or move the model to the registry
        if os.path.isdir(model_path):
            # Copy the directory contents
            for item in os.listdir(model_path):
                s = os.path.join(model_path, item)
                d = os.path.join(model_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            registered_model_path = model_dir
        else:
            # Copy the file
            registered_model_path = os.path.join(model_dir, os.path.basename(model_path))
            shutil.copy2(model_path, registered_model_path)
            
        timestamp = datetime.datetime.now().isoformat()
        
        # Create and save model version info
        model_info = {
            "version": version,
            "created_at": timestamp,
            "path": os.path.relpath(registered_model_path, self.registry_dir),
            "description": description or "",
            "tags": tags or [],
            "metadata": metadata or {}
        }
        
        # Update registry
        registry["models"][name]["versions"][version] = model_info
        
        # If this is the first version, set it as the latest
        if len(registry["models"][name]["versions"]) == 1:
            registry["models"][name]["latest_version"] = version
            
        # Save updated registry
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
            
        return model_info
    
    def get_model(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model info from the registry.
        
        Args:
            name: Name of the model
            version: Specific version to get, if None returns the latest
            
        Returns:
            Model info dictionary
        """
        with open(self.registry_file, 'r') as f:
            registry = json.load(f)
            
        if name not in registry["models"]:
            raise ValueError(f"Model not found: {name}")
            
        model_data = registry["models"][name]
        
        # Determine which version to return
        if version is None:
            version = model_data.get("latest_version")
            if version is None:
                raise ValueError(f"No latest version set for model: {name}")
        elif version not in model_data["versions"]:
            raise ValueError(f"Version not found for model {name}: {version}")
            
        model_info = model_data["versions"][version]
        
        # Add the absolute path
        model_info["absolute_path"] = os.path.join(self.registry_dir, model_info["path"])
        
        return model_info
    
    def set_latest_version(self, name: str, version: str) -> None:
        """Set a specific version as the latest for a model."""
        with open(self.registry_file, 'r') as f:
            registry = json.load(f)
            
        if name not in registry["models"]:
            raise ValueError(f"Model not found: {name}")
            
        if version not in registry["models"][name]["versions"]:
            raise ValueError(f"Version not found for model {name}: {version}")
            
        registry["models"][name]["latest_version"] = version
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
            
    def list_models(self) -> Dict[str, Any]:
        """List all models in the registry."""
        with open(self.registry_file, 'r') as f:
            registry = json.load(f)
            
        # Create a more concise version for listing
        models_list = {}
        for model_name, model_data in registry["models"].items():
            versions = list(model_data["versions"].keys())
            latest = model_data.get("latest_version")
            models_list[model_name] = {
                "versions": versions,
                "latest_version": latest,
                "version_count": len(versions)
            }
            
        return models_list
    
    def compare_models(self, name: str, version1: str, version2: str, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare two versions of a model.
        
        Args:
            name: Name of the model
            version1: First version to compare
            version2: Second version to compare
            metrics: List of metric names to compare, if None compares all
            
        Returns:
            Comparison results
        """
        model1 = self.get_model(name, version1)
        model2 = self.get_model(name, version2)
        
        comparison = {
            "model_name": name,
            "version1": version1,
            "version2": version2,
            "metrics_comparison": {}
        }
        
        # Compare metrics if available
        metrics1 = model1.get("metadata", {}).get("metrics", {})
        metrics2 = model2.get("metadata", {}).get("metrics", {})
        
        if metrics is None:
            # Compare all metrics that exist in both versions
            metrics = set(metrics1.keys()) & set(metrics2.keys())
            
        for metric in metrics:
            if metric in metrics1 and metric in metrics2:
                value1 = metrics1[metric]
                value2 = metrics2[metric]
                comparison["metrics_comparison"][metric] = {
                    "version1": value1,
                    "version2": value2,
                    "difference": value2 - value1,
                    "percent_change": ((value2 - value1) / value1 * 100) if value1 != 0 else float('inf')
                }
                
        return comparison


# core/pipeline_manager.py
from typing import Dict, Any, List, Optional, Callable, Type
import importlib
import inspect
import os
import json
import datetime

class PipelineManager:
    def __init__(self, pipelines_dir: str = "pipelines"):
        self.pipelines_dir = pipelines_dir
        self.registered_pipelines = {}
        self._initialize()
        
    def _initialize(self):
        """Initialize the pipeline manager by scanning for pipeline classes."""
        # Ensure pipelines directory exists
        os.makedirs(self.pipelines_dir, exist_ok=True)
        
        # Create a metadata file for tracking pipelines if it doesn't exist
        metadata_file = os.path.join(self.pipelines_dir, "pipelines_metadata.json")
        if not os.path.exists(metadata_file):
            with open(metadata_file, 'w') as f:
                json.dump({"pipelines": {}}, f, indent=2)
                
    def register_pipeline(self, pipeline_class: Type, name: Optional[str] = None) -> str:
        """
        Register a pipeline class.
        
        Args:
            pipeline_class: A class implementing the pipeline interface
            name: Optional name for the pipeline (defaults to class name)
            
        Returns:
            The registered pipeline name
        """
        if name is None:
            name = pipeline_class.__name__
            
        # Store the pipeline class
        self.registered_pipelines[name] = pipeline_class
        
        # Update metadata file
        metadata_file = os.path.join(self.pipelines_dir, "pipelines_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        # Add pipeline if not already registered
        if name not in metadata["pipelines"]:
            metadata["pipelines"][name] = {
                "class": pipeline_class.__name__,
                "module": pipeline_class.__module__,
                "registered_at": datetime.datetime.now().isoformat(),
                "runs": []
            }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return name
    
    def get_pipeline(self, name: str) -> Type:
        """
        Get a registered pipeline class by name.
        
        Args:
            name: Name of the pipeline
            
        Returns:
            The pipeline class
        """
        if name in self.registered_pipelines:
            return self.registered_pipelines[name]
            
        # Try to load the pipeline from metadata
        metadata_file = os.path.join(self.pipelines_dir, "pipelines_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        if name in metadata["pipelines"]:
            pipeline_info = metadata["pipelines"][name]
            module_name = pipeline_info["module"]
            class_name = pipeline_info["class"]
            
            try:
                module = importlib.import_module(module_name)
                pipeline_class = getattr(module, class_name)
                self.registered_pipelines[name] = pipeline_class
                return pipeline_class
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to load pipeline '{name}': {e}")
        
        raise ValueError(f"Pipeline not found: {name}")
    
    def execute_pipeline(self, 
                       name: str, 
                       params: Optional[Dict[str, Any]] = None, 
                       artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a registered pipeline.
        
        Args:
            name: Name of the pipeline to execute
            params: Parameters to pass to the pipeline
            artifacts: Input artifacts for the pipeline
            
        Returns:
            The pipeline execution results
        """
        pipeline_class = self.get_pipeline(name)
        
        # Instantiate and run the pipeline
        pipeline = pipeline_class(params=params or {})
        results = pipeline.run(artifacts=artifacts or {})
        
        # Log the pipeline run
        run_id = self._log_pipeline_run(name, params, results)
        
        return {
            "run_id": run_id,
            "results": results
        }
    
    def list_pipelines(self) -> List[str]:
        """
        List all registered pipelines.
        
        Returns:
            List of pipeline names
        """
        metadata_file = os.path.join(self.pipelines_dir, "pipelines_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        return list(metadata["pipelines"].keys())
    
    def _log_pipeline_run(self, 
                        name: str, 
                        params: Optional[Dict[str, Any]], 
                        results: Dict[str, Any]) -> str:
        """
        Log a pipeline run.
        
        Args:
            name: Pipeline name
            params: Parameters used
            results: Execution results
            
        Returns:
            Run ID
        """
        metadata_file = os.path.join(self.pipelines_dir, "pipelines_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        if name not in metadata["pipelines"]:
            raise ValueError(f"Pipeline not found: {name}")
            
        # Create a run ID
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(metadata['pipelines'][name]['runs'])}"
        
        # Create run record
        run_record = {
            "id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "params": params or {},
            "result_summary": {
                # Include only non-artifact results
                k: v for k, v in results.items() 
                if not isinstance(v, (bytes, bytearray)) and 
                   not k.startswith("artifact_")
            }
        }
        
        # Add to metadata
        metadata["pipelines"][name]["runs"].append(run_record)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)