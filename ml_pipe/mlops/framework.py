import os
import uuid
import json
import yaml
import pickle
import logging
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# -----------------------------
# Configuration
# -----------------------------

class Config:
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key with optional default."""
        if "." in key:
            # Support nested access with dot notation
            parts = key.split(".")
            current = self.config_dict
            for part in parts[:-1]:
                if part not in current:
                    return default
                current = current[part]
            return current.get(parts[-1], default)
        return self.config_dict.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self.config_dict[key] = value

    @staticmethod
    def load_file(config_path: str) -> "Config":
        """Loads configuration from a YAML or JSON file."""
        try:
            with open(config_path, "r") as file:
                if config_path.endswith(('.yml', '.yaml')):
                    config_data = yaml.safe_load(file)
                elif config_path.endswith('.json'):
                    config_data = json.load(file)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path}")
            return Config(config_data)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")

    def save_file(self, config_path: str) -> None:
        """Saves configuration to a YAML or JSON file."""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as file:
                if config_path.endswith(('.yml', '.yaml')):
                    yaml.dump(self.config_dict, file)
                elif config_path.endswith('.json'):
                    json.dump(self.config_dict, file, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path}")
        except Exception as e:
            raise ValueError(f"Error saving config file {config_path}: {e}")


# -----------------------------
# Artifact Store
# -----------------------------

class ArtifactStore:
    """Stores and retrieves intermediate artifacts for the pipeline."""

    def __init__(self, config: Optional[Config] = None, run_id: str = None):
        self.config = config or Config()
        self.run_id = run_id or str(uuid.uuid4())
        self._artifacts: Dict[str, Any] = {}
        
        # Set up base path with run_id for versioning
        base_dir = self.config.get("folder_path.artifacts", "artifacts")
        self.base_path = os.path.join(base_dir, self.run_id)
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Artifact store initialized at '{self.base_path}'")

    def save_artifact(self, artifact: Any, subdir: str, name: str) -> str:
        """Save an artifact to disk and return its path."""
        artifact_dir = os.path.join(self.base_path, subdir)
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, name)

        try:
            if name.endswith(".pkl"):
                with open(artifact_path, "wb") as f:
                    pickle.dump(artifact, f)
            elif name.endswith(".csv"):
                if hasattr(artifact, "to_csv"):
                    artifact.to_csv(artifact_path, index=False)
                else:
                    raise TypeError(f"Object of type {type(artifact)} cannot be saved as CSV")
            elif name.endswith(".txt"):
                with open(artifact_path, "w") as f:
                    f.write(str(artifact))
            elif name.endswith(".json"):
                with open(artifact_path, "w") as f:
                    if isinstance(artifact, dict) or isinstance(artifact, list):
                        json.dump(artifact, f, indent=2)
                    else:
                        json.dump(str(artifact), f)
            else:
                raise ValueError(f"Unsupported format for {name}")
                
            logging.info(f"Artifact '{name}' saved to {artifact_path}")
            
            # Also store in memory for quick access
            key = f"{subdir}/{name}"
            self._artifacts[key] = artifact
            
            return artifact_path
        except Exception as e:
            logging.error(f"Failed to save artifact '{name}': {str(e)}")
            raise

    def load_artifact(self, subdir: str, name: str) -> Optional[Any]:
        """Load an artifact from disk."""
        # Check if it's already in memory
        key = f"{subdir}/{name}"
        if key in self._artifacts:
            return self._artifacts[key]
            
        # Otherwise load from disk
        artifact_path = os.path.join(self.base_path, subdir, name)
        if os.path.exists(artifact_path):
            try:
                if name.endswith(".pkl"):
                    with open(artifact_path, "rb") as f:
                        artifact = pickle.load(f)
                elif name.endswith(".csv"):
                    artifact = pd.read_csv(artifact_path)
                elif name.endswith(".txt"):
                    with open(artifact_path, "r") as f:
                        artifact = f.read()
                elif name.endswith(".json"):
                    with open(artifact_path, "r") as f:
                        artifact = json.load(f)
                else:
                    raise ValueError(f"Unsupported format for {name}")
                
                logging.info(f"Artifact '{name}' loaded from {artifact_path}")
                
                # Cache in memory
                self._artifacts[key] = artifact
                
                return artifact
            except Exception as e:
                logging.error(f"Failed to load artifact '{name}': {str(e)}")
                raise
        else:
            logging.warning(f"Artifact '{name}' not found at {artifact_path}")
            return None

    def store(self, key: str, data: Any) -> str:
        """Store an artifact in memory and return its key."""
        self._artifacts[key] = data
        return key
    
    def get_artifact(self, key: str) -> Optional[Any]:
        """Retrieve an in-memory artifact by key."""
        return self._artifacts.get(key)
        
    def list_artifacts(self, run_id: Optional[str] = None) -> List[str]:
        """List all artifacts for a given run."""
        target_path = os.path.join(
            self.config.get("folder_path.artifacts", "artifacts"),
            run_id or self.run_id
        )
        
        if not os.path.exists(target_path):
            return []
            
        artifacts = []
        for root, _, files in os.walk(target_path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), target_path)
                artifacts.append(rel_path)
        
        return artifacts


# -----------------------------
# Task Class
# -----------------------------

class Task:
    """Base class for pipeline tasks."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.status = "pending"
        self.start_time = None
        self.end_time = None
        self.result = None
        self.error = None
        
    def execute(self, stack: "StackPipeline", context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the task and return updated context."""
        self.start_time = datetime.now()
        self.status = "running"
        
        try:
            logging.info(f"Starting task: {self.name}")
            updated_context = self._run(stack, context)
            self.status = "completed"
            self.result = updated_context.get("result", None)
            return updated_context
            
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            logging.error(f"Task '{self.name}' failed: {str(e)}")
            raise
            
        finally:
            self.end_time = datetime.now()
            
    def _run(self, stack: "StackPipeline", context: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation to be provided by subclasses."""
        raise NotImplementedError("Subclasses must implement _run method")


# -----------------------------
# StackPipeline
# -----------------------------

class StackPipeline:
    """A stack that brings together all components needed to run pipelines."""
    
    def __init__(self, name: str, description: str = "", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.config = Config(config or {})
        self.artifact_store = ArtifactStore(self.config)
        self.tasks = []
        self.components = {}
        self._run_history = {}

    def add_task(self, task: Task) -> "StackPipeline":
        """Add a task to the pipeline."""
        self.tasks.append(task)
        return self
    
    def register_component(self, name: str, component: Any) -> None:
        """Register a component to the stack."""
        self.components[name] = component
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get a registered component by name."""
        return self.components.get(name)
    
    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Load run details from artifacts."""
        if run_id in self._run_history:
            return self._run_history[run_id]
        return self.artifact_store.load_artifact("metadata", "run_info.json") or {}

    def run(self, initial_context: Dict[str, Any] = None, tags: Dict[str, str] = None) -> str:
        """Execute the pipeline with the given context and tags."""
        run_id = str(uuid.uuid4())
        self.artifact_store.run_id = run_id
        
        context = initial_context or {}
        start_time = datetime.now()
        status = "running"
        
        run_info = {
            "run_id": run_id,
            "pipeline_name": self.name,
            "start_time": start_time.isoformat(),
            "end_time": None,
            "status": status,
            "tags": tags or {},
            "tasks": []
        }
        
        try:
            for task in self.tasks:
                logging.info(f"Running task: {task.name}")
                context = task.execute(self, context)
                run_info["tasks"].append({
                    "name": task.name,
                    "status": task.status,
                    "start_time": task.start_time.isoformat() if task.start_time else None,
                    "end_time": task.end_time.isoformat() if task.end_time else None
                })
            
            status = "completed"
        except Exception as e:
            logging.error(f"Pipeline '{self.name}' failed: {str(e)}")
            status = "failed"
            raise
        finally:
            end_time = datetime.now()
            run_info["end_time"] = end_time.isoformat()
            run_info["status"] = status
            run_info["duration_seconds"] = (end_time - start_time).total_seconds()
            self._run_history[run_id] = run_info
            
            # Save run metadata to artifacts
            self.artifact_store.save_artifact(
                run_info, 
                subdir="metadata", 
                name="run_info.json"
            )
            
        return run_id


# ----------------------------- 
# Data Ingestion Task
# -----------------------------

class DataIngestionTask(Task):
    """Task for loading and splitting data."""

    def __init__(self, data_path: str):
        super().__init__(name="data_ingestion", description="Load and split data")
        self.data_path = data_path
        
    def _run(self, stack: StackPipeline, context: Dict[str, Any]) -> Dict[str, Any]:
        # Load raw data
        df = pd.read_csv(self.data_path)
        logging.info(f"Loaded data from {self.data_path}, shape: {df.shape}")
        
        # Save the full dataset as an artifact
        stack.artifact_store.save_artifact(
            df, 
            subdir="raw", 
            name="full_dataset.csv"
        )

        # Split data
        test_size = stack.config.get("base.test_size", 0.2)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        logging.info(f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}")

        # Save split datasets
        stack.artifact_store.save_artifact(
            train_data, 
            subdir="raw", 
            name="train_data.csv" 
        )

        stack.artifact_store.save_artifact(
            test_data, 
            subdir="raw", 
            name="test_data.csv"
        )

        # Update context with data
        updated_context = {**context}
        updated_context["train_data"] = train_data
        updated_context["test_data"] = test_data
        
        return updated_context


# ----------------------------- 
# Data Processing Task
# -----------------------------

class DataProcessingTask(Task):
    """Task for preprocessing data."""

    def __init__(self):
        super().__init__(name="data_processing", description="Preprocess and transform data")
        
    def _run(self, stack: StackPipeline, context: Dict[str, Any]) -> Dict[str, Any]:
        train_data = context.get("train_data")
        test_data = context.get("test_data")
        
        if train_data is None or test_data is None:
            raise ValueError("Train or test data not found in context. Run data ingestion first.")
            
        # Example preprocessing (replace with actual preprocessing logic)
        # This could include feature engineering, scaling, encoding, etc.
        train_processed = train_data.copy()
        test_processed = test_data.copy()
        
        # Handle missing values
        for col in train_processed.columns:
            if train_processed[col].dtype.kind in 'ifc':  # integer, float, complex
                # Replace numeric missing values with mean
                mean_val = train_processed[col].mean()
                train_processed[col] = train_processed[col].fillna(mean_val)
                test_processed[col] = test_processed[col].fillna(mean_val)
            else:
                # Replace categorical missing values with most frequent
                mode_val = train_processed[col].mode().iloc[0] if not train_processed[col].mode().empty else "unknown"
                train_processed[col] = train_processed[col].fillna(mode_val)
                test_processed[col] = test_processed[col].fillna(mode_val)
        
        logging.info("Data preprocessing completed")
        
        # Save processed datasets
        stack.artifact_store.save_artifact(
            train_processed,
            subdir="processed",
            name="train_processed.csv"
        )
        
        stack.artifact_store.save_artifact(
            test_processed,
            subdir="processed",
            name="test_processed.csv"
        )
        
        # Update context
        updated_context = {**context}
        updated_context["train_processed"] = train_processed
        updated_context["test_processed"] = test_processed
        
        return updated_context


# ----------------------------- 
# Model Training Task
# -----------------------------

class ModelTrainingTask(Task):
    """Task for training a model."""

    def __init__(self, model_type: str = "default"):
        super().__init__(name="model_training", description=f"Train {model_type} model")
        self.model_type = model_type
        
    def _run(self, stack: StackPipeline, context: Dict[str, Any]) -> Dict[str, Any]:
        train_processed = context.get("train_processed")
        
        if train_processed is None:
            raise ValueError("Processed training data not found in context. Run data processing first.")
        
        # Here you would implement the actual model training
        # This is a placeholder for demonstration purposes
        
        # Example:
        # from sklearn.ensemble import RandomForestClassifier
        # 
        # target_col = stack.config.get("model.target_column", "target")
        # X = train_processed.drop(columns=[target_col])
        # y = train_processed[target_col]
        #
        # model = RandomForestClassifier()
        # model.fit(X, y)
        
        # For now, just create a dummy model object
        model = {"type": self.model_type, "trained": True, "timestamp": datetime.now().isoformat()}
        
        # Save the model
        stack.artifact_store.save_artifact(
            model,
            subdir="models",
            name="trained_model.pkl"
        )
        
        logging.info(f"Model training completed for {self.model_type} model")
        
        # Update context
        updated_context = {**context}
        updated_context["model"] = model
        
        return updated_context


# ----------------------------- 
# Model Evaluation Task
# -----------------------------

class ModelEvaluationTask(Task):
    """Task for evaluating a trained model."""

    def __init__(self):
        super().__init__(name="model_evaluation", description="Evaluate model performance")
        
    def _run(self, stack: StackPipeline, context: Dict[str, Any]) -> Dict[str, Any]:
        model = context.get("model")
        test_processed = context.get("test_processed")
        
        if model is None or test_processed is None:
            raise ValueError("Model or test data not found in context. Run model training first.")
        
        # Here you would implement the actual model evaluation
        # This is a placeholder for demonstration purposes
        
        # Example:
        # target_col = stack.config.get("model.target_column", "target")
        # X_test = test_processed.drop(columns=[target_col])
        # y_test = test_processed[target_col]
        #
        # y_pred = model.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)
        
        # For now, just create dummy metrics
        metrics = {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.81,
            "f1_score": 0.82
        }
        
        # Save the metrics
        stack.artifact_store.save_artifact(
            metrics,
            subdir="metrics",
            name="evaluation_metrics.json"
        )
        
        logging.info(f"Model evaluation completed with accuracy: {metrics['accuracy']}")
        
        # Update context
        updated_context = {**context}
        updated_context["metrics"] = metrics
        
        return updated_context


# ----------------------------- 
# Training Pipeline
# -----------------------------

class TrainingPipeline:
    """Main pipeline class that orchestrates the training workflow."""

    def __init__(self, data_path: str, config_path: str = "config/config.yml"):
        self.data_path = data_path
        
        # Load configuration
        try:
            self.config = Config.load_file(config_path).config_dict
            logging.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {str(e)}. Using default configuration.")
            self.config = {}
        
    def run(self) -> Tuple[str, StackPipeline]:
        """Execute the training pipeline."""
        # Create stack
        stack = StackPipeline(name="training_pipeline", config=self.config)
        
        # Add tasks to pipeline
        stack.add_task(DataIngestionTask(self.data_path))
        stack.add_task(DataProcessingTask())
        stack.add_task(ModelTrainingTask(model_type=stack.config.get("model.type", "default")))
        stack.add_task(ModelEvaluationTask())
        
        # Run the pipeline
        try:
            run_id = stack.run()
            logging.info(f"Pipeline completed successfully with run ID: {run_id}")
            
            # Output run summary
            self._print_run_summary(stack, run_id)
            
            return run_id, stack
            
        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}")
            raise
            
    def _print_run_summary(self, stack: StackPipeline, run_id: str) -> None:
        """Print a summary of the pipeline run."""
        # List artifacts
        artifacts = stack.artifact_store.list_artifacts(run_id)
        print(f"\nRun ID: {run_id}")
        print("\nArtifacts:")
        for uri in artifacts:
            print(f"- {uri}")
            
        # Get run details
        run_details = stack.get_run_details(run_id)
        print("\nRun Details:")
        print(f"Pipeline: {run_details.get('pipeline_name')}")
        print(f"Status: {run_details.get('status')}")
        print(f"Duration: {run_details.get('duration_seconds', 0):.2f} seconds")
        
        # Check if run was successful
        if run_details.get("status") == "completed":
            print("Pipeline completed successfully")
            
            # Print evaluation metrics if available
            metrics = stack.artifact_store.load_artifact("metrics", "evaluation_metrics.json")
            if metrics:
                print("\nEvaluation Metrics:")
                for metric, value in metrics.items():
                    print(f"- {metric}: {value:.4f}")


# ----------------------------- 
# Example Usage
# ----------------------------- 

def main():
    """Main entry point for running the pipeline."""
    # Path to your data file
    data_path = "data.csv"
    
    # Create and run the pipeline
    try:
        pipeline = TrainingPipeline(data_path)
        run_id, stack = pipeline.run()
        
        print("\n" + "="*50)
        print("Pipeline execution complete!")
        print(f"Run ID: {run_id}")
        print("="*50)
        
        return run_id, stack
        
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")
        return None, None


if __name__ == "__main__":
    main()
