import os
import uuid
import yaml
import json
import pickle
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------

class Config:
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key with optional default."""
        return self.config_dict.get(key, default)

    @staticmethod
    def load_file(config_path: str) -> "Config":
        """Loads configuration from a YAML or JSON file."""
        try:
            with open(config_path, "r") as file:
                if config_path.endswith(('.yml', '.yaml')):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
            return Config(config_data)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")


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
        base_dir = self.config.get("folder_path", {}).get("artifacts", "artifacts")
        self.base_path = os.path.join(base_dir, self.run_id)
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Artifact store initialized at '{self.base_path}'")

    def save_artifact(self, artifact: Any, subdir: str, name: str) -> str:
        """Save an artifact to disk and return its path."""
        artifact_dir = os.path.join(self.base_path, subdir)
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, name)

        if name.endswith(".pkl"):
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact, f)
        elif name.endswith((".csv", ".txt")):
            if hasattr(artifact, "to_csv"):
                artifact.to_csv(artifact_path, index=False)
            else:
                with open(artifact_path, "w") as f:
                    f.write(str(artifact))
        elif name.endswith((".json")):
            with open(artifact_path, "w") as f:
                json.dump(artifact, f)
        else:
            raise ValueError(f"Unsupported format for {name}")
            
        logging.info(f"Artifact '{name}' saved to {artifact_path}")
        return artifact_path

    def load_artifact(self, subdir: str, name: str) -> Optional[Any]:
        """Load an artifact from disk."""
        artifact_path = os.path.join(self.base_path, subdir, name)
        if os.path.exists(artifact_path):
            if name.endswith(".pkl"):
                with open(artifact_path, "rb") as f:
                    artifact = pickle.load(f)
            elif name.endswith(".csv"):
                artifact = pd.read_csv(artifact_path)
            elif name.endswith(".json"):
                with open(artifact_path, "r") as f:
                    artifact = json.load(f)
            else:
                raise ValueError(f"Unsupported format for {name}")
            logging.info(f"Artifact '{name}' loaded from {artifact_path}")
            return artifact
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
            self.config.get("folder_path", {}).get("artifacts", "artifacts"),
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
# Pipeline Steps
# -----------------------------

class PipelineStep:
    """Base class for all pipeline steps."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.start_time = None
        self.end_time = None
        self.status = "pending"
        
    def execute(self, stack: "Stack", context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline step.
        
        Args:
            stack: The stack instance containing all resources
            context: The context data from previous steps
            
        Returns:
            Updated context with step outputs
        """
        self.start_time = datetime.now()
        self.status = "running"
        
        try:
            outputs = self._run(stack, context)
            self.status = "completed"
            return {**context, **outputs}
        except Exception as e:
            self.status = "failed"
            logging.error(f"Step {self.name} failed: {str(e)}")
            raise
        finally:
            self.end_time = datetime.now()
    
    def _run(self, stack: "Stack", context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the actual step logic, to be implemented by subclasses.
        
        Args:
            stack: The stack instance
            context: The context data
            
        Returns:
            Dict of outputs to add to context
        """
        raise NotImplementedError("Subclasses must implement _run")


# -----------------------------
# Pipeline
# -----------------------------

class Pipeline:
    """A pipeline of consecutive steps to be executed."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: List[PipelineStep] = []
        self._run_history: Dict[str, Dict[str, Any]] = {}
    
    def add_step(self, step: PipelineStep) -> "Pipeline":
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self
        
    def run(self, stack: "Stack", initial_context: Dict[str, Any] = None, tags: Dict[str, str] = None) -> str:
        """
        Run the pipeline with the given stack and context.
        
        Args:
            stack: The stack instance containing all resources
            initial_context: Initial context data
            tags: Optional tags for the run
            
        Returns:
            run_id: The unique identifier for this run
        """
        run_id = str(uuid.uuid4())
        stack.artifact_store.run_id = run_id
        
        context = initial_context or {}
        start_time = datetime.now()
        status = "running"
        
        run_info = {
            "run_id": run_id,
            "pipeline_name": self.name,
            "start_time": start_time,
            "end_time": None,
            "status": status,
            "tags": tags or {},
            "steps": []
        }
        
        try:
            for step in self.steps:
                logging.info(f"Running step: {step.name}")
                context = step.execute(stack, context)
                run_info["steps"].append({
                    "name": step.name,
                    "status": step.status,
                    "start_time": step.start_time,
                    "end_time": step.end_time
                })
            
            status = "completed"
        except Exception as e:
            logging.error(f"Pipeline '{self.name}' failed: {str(e)}")
            status = "failed"
            raise
        finally:
            end_time = datetime.now()
            run_info["end_time"] = end_time
            run_info["status"] = status
            self._run_history[run_id] = run_info
            
            # Save run metadata to artifacts
            stack.artifact_store.save_artifact(
                run_info, 
                subdir="metadata", 
                name=f"run_info.json"
            )
            
        return run_id
    
    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get details for a specific run."""
        return self._run_history.get(run_id, {})


# -----------------------------
# Stack
# -----------------------------

class Stack:
    """A stack that brings together all components needed to run pipelines."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = Config(config or {})
        self.artifact_store = ArtifactStore(self.config)
        self.components = {}
    
    def register_component(self, name: str, component: Any) -> None:
        """Register a component to the stack."""
        self.components[name] = component
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get a registered component by name."""
        return self.components.get(name)
    
    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Load run details from artifacts."""
        return self.artifact_store.load_artifact("metadata", "run_info.json") or {}


# ----------------------------- 
# Data Ingestion
# ----------------------------- 

class DataIngestion(PipelineStep):
    def __init__(self, input_path: str, raw_path: str = "raw", 
                 train_filename: str = "train.pkl", test_filename: str = "test.pkl"):
        super().__init__(name="data_ingestion", 
                         description="Load and split data into train and test sets")
        self.input_path = input_path
        self.raw_path = raw_path
        self.train_filename = train_filename
        self.test_filename = test_filename
        
    def _run(self, stack: Stack, context: Dict[str, Any]) -> Dict[str, Any]:
        logging.info(f"Ingesting data from {self.input_path}")
        
        # Load raw data
        df = pd.read_csv(self.input_path)
        
        # Split data
        test_size = stack.config.get("base", {}).get("test_size", 0.2)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        # Save artifacts
        stack.artifact_store.save_artifact(
            train_data, subdir=self.raw_path, name=self.train_filename
        )
        stack.artifact_store.save_artifact(
            test_data, subdir=self.raw_path, name=self.test_filename
        )
        
        return {
            "train_data": train_data,
            "test_data": test_data,
            "train_path": f"{self.raw_path}/{self.train_filename}",
            "test_path": f"{self.raw_path}/{self.test_filename}"
        }


# ----------------------------- 
# Data Preprocessing
# ----------------------------- 

class DataPreprocessor(PipelineStep):
    def __init__(self, processed_path: str = "processed", 
                 train_filename: str = "processed_train.pkl", 
                 test_filename: str = "processed_test.pkl"):
        super().__init__(name="data_preprocessing", 
                         description="Preprocess and transform raw data")
        self.processed_path = processed_path
        self.train_filename = train_filename
        self.test_filename = test_filename
        
    def _run(self, stack: Stack, context: Dict[str, Any]) -> Dict[str, Any]:
        # Get train and test data from context
        train_data = context.get("train_data")
        test_data = context.get("test_data")
        
        if train_data is None or test_data is None:
            # Try to load from artifact store
            train_path = context.get("train_path", "raw/train.pkl")
            test_path = context.get("test_path", "raw/test.pkl")
            
            train_subdir, train_name = os.path.split(train_path)
            test_subdir, test_name = os.path.split(test_path)
            
            train_data = stack.artifact_store.load_artifact(train_subdir, train_name)
            test_data = stack.artifact_store.load_artifact(test_subdir, test_name)
        
        if train_data is None or test_data is None:
            raise ValueError("Train or test data not found in context or artifacts")
        
        # Perform preprocessing (customize as needed)
        processed_train = self._preprocess(train_data, is_training=True)
        processed_test = self._preprocess(test_data, is_training=False)
        
        # Save processed data
        stack.artifact_store.save_artifact(
            processed_train, subdir=self.processed_path, name=self.train_filename
        )
        stack.artifact_store.save_artifact(
            processed_test, subdir=self.processed_path, name=self.test_filename
        )
        
        return {
            "processed_train": processed_train,
            "processed_test": processed_test,
            "processed_train_path": f"{self.processed_path}/{self.train_filename}",
            "processed_test_path": f"{self.processed_path}/{self.test_filename}"
        }
    
    def _preprocess(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Apply preprocessing transformations to the data.
        Override this method to implement custom preprocessing logic.
        """
        # Example preprocessing (customize as needed)
        df_processed = df.copy()
        
        # Handle missing values
        for col in df_processed.columns:
            if df_processed[col].dtype.kind in 'ifc':  # integer, float or complex
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        
        return df_processed


# ----------------------------- 
# Model Training
# ----------------------------- 

class ModelTrainer(PipelineStep):
    def __init__(self, model_path: str = "models", model_filename: str = "model.pkl"):
        super().__init__(name="model_training", 
                         description="Train machine learning model")
        self.model_path = model_path
        self.model_filename = model_filename
        
    def _run(self, stack: Stack, context: Dict[str, Any]) -> Dict[str, Any]:
        # Get processed train data
        train_data = context.get("processed_train")
        
        if train_data is None:
            # Try to load from artifact store
            train_path = context.get("processed_train_path", "processed/processed_train.pkl")
            train_subdir, train_name = os.path.split(train_path)
            train_data = stack.artifact_store.load_artifact(train_subdir, train_name)
        
        if train_data is None:
            raise ValueError("Processed train data not found in context or artifacts")
        
        # Define features and target
        feature_cols = stack.config.get("model", {}).get("feature_columns", [])
        target_col = stack.config.get("model", {}).get("target_column", "target")
        
        if not feature_cols:
            # Use all columns except target as features
            feature_cols = [col for col in train_data.columns if col != target_col]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        
        # Train model (customize as needed)
        model = self._train_model(X_train, y_train, stack.config)
        
        # Save model
        stack.artifact_store.save_artifact(
            model, subdir=self.model_path, name=self.model_filename
        )
        
        # Save feature columns for inference
        feature_metadata = {
            "feature_columns": feature_cols,
            "target_column": target_col
        }
        
        stack.artifact_store.save_artifact(
            feature_metadata, subdir=self.model_path, name="feature_metadata.json"
        )
        
        return {
            "model": model,
            "model_path": f"{self.model_path}/{self.model_filename}",
            "feature_metadata": feature_metadata
        }
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, config: Config) -> Any:
        """
        Train a model on the given data.
        Override this method to implement custom training logic.
        """
        # This is a placeholder - you'd implement actual model training here
        # For example, you might use scikit-learn:
        from sklearn.ensemble import RandomForestClassifier
        
        model_type = config.get("model", {}).get("type", "RandomForestClassifier")
        
        if model_type == "RandomForestClassifier":
            model = RandomForestClassifier(
                n_estimators=config.get("model", {}).get("n_estimators", 100),
                max_depth=config.get("model", {}).get("max_depth", None),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        model.fit(X, y)
        return model


# ----------------------------- 
# Model Evaluation
# ----------------------------- 

class ModelEvaluator(PipelineStep):
    def __init__(self, metrics_path: str = "metrics"):
        super().__init__(name="model_evaluation", 
                         description="Evaluate model performance")
        self.metrics_path = metrics_path
        
    def _run(self, stack: Stack, context: Dict[str, Any]) -> Dict[str, Any]:
        # Get model and test data
        model = context.get("model")
        test_data = context.get("processed_test")
        feature_metadata = context.get("feature_metadata")
        
        if model is None:
            # Try to load from artifact store
            model_path = context.get("model_path", "models/model.pkl")
            model_subdir, model_name = os.path.split(model_path)
            model = stack.artifact_store.load_artifact(model_subdir, model_name)
            
        if test_data is None:
            # Try to load from artifact store
            test_path = context.get("processed_test_path", "processed/processed_test.pkl")
            test_subdir, test_name = os.path.split(test_path)
            test_data = stack.artifact_store.load_artifact(test_subdir, test_name)
            
        if feature_metadata is None:
            # Try to load from artifact store
            feature_metadata = stack.artifact_store.load_artifact("models", "feature_metadata.json")
            
        if model is None or test_data is None or feature_metadata is None:
            raise ValueError("Model, test data, or feature metadata not found")
            
        # Extract feature and target columns
        feature_cols = feature_metadata.get("feature_columns", [])
        target_col = feature_metadata.get("target_column", "target")
        
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        # Evaluate model
        metrics = self._evaluate_model(model, X_test, y_test)
        
        # Save metrics
        stack.artifact_store.save_artifact(
            metrics, subdir=self.metrics_path, name="evaluation_metrics.json"
        )
        
        return {
            "evaluation_metrics": metrics,
            "metrics_path": f"{self.metrics_path}/evaluation_metrics.json"
        }
    
    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model and return metrics.
        Override this method to implement custom evaluation logic.
        """
        # This is a placeholder - you'd implement actual evaluation here
        # For example:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = model.predict(X)
        
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y, y_pred, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, average='weighted', zero_division=0))
        }
        
        return metrics


# ----------------------------- 
# TrainingPipeline
# -----------------------------

class TrainingPipeline:
    """Main pipeline class that orchestrates the training workflow."""

    def __init__(self, path: str, config_path: str = "config/config.yml"):
        self.path = path
        self.config = Config.load_file(config_path).config_dict
        
    def create_stack(self) -> Stack:
        """Create and configure the stack for this pipeline."""
        stack = Stack(name="training_pipeline", config=self.config)
        return stack
        
    def create_pipeline(self) -> Pipeline:
        """Create the pipeline with all required steps."""
        pipeline = Pipeline(
            name="training_pipeline", 
            description="End-to-end training pipeline"
        )
        
        # Add pipeline steps
        pipeline.add_step(DataIngestion(self.path))
        pipeline.add_step(DataPreprocessor())
        pipeline.add_step(ModelTrainer())
        pipeline.add_step(ModelEvaluator())
        
        return pipeline
        
    def run(self) -> Tuple[str, Stack]:
        """Execute the training pipeline."""
        # Create stack and pipeline
        stack = self.create_stack()
        pipeline = self.create_pipeline()
        
        # Run the pipeline
        run_id = pipeline.run(stack, tags={"env": "dev"})
        
        # List artifacts
        artifacts = stack.artifact_store.list_artifacts(run_id)
        print(f"Run ID: {run_id}")
        print("Artifacts:")
        for uri in artifacts:
            print(f"- {uri}")
            
        # Get run details
        run_details = stack.get_run_details(run_id)
        print("\nRun Details:")
        print(f"Pipeline: {run_details.get('pipeline_name')}")
        print(f"Status: {run_details.get('status')}")
        
        # Check if run was successful
        if run_details.get("status") == "completed":
            print("Pipeline completed successfully")
            
            # Print evaluation metrics if available
            metrics = stack.artifact_store.load_artifact("metrics", "evaluation_metrics.json")
            if metrics:
                print("\nEvaluation Metrics:")
                for metric, value in metrics.items():
                    print(f"- {metric}: {value:.4f}")
        
        return run_id, stack


# ----------------------------- 
# Example Usage
# ----------------------------- 

def main():
    # Path to your data file
    data_path = "data.csv"
    
    # Create and run the pipeline
    pipeline = TrainingPipeline(data_path)
    run_id, stack = pipeline.run()
    
    return run_id, stack


if __name__ == "__main__":
    main()
