import os
import logging
import json
import yaml
import pickle
import uuid
import inspect
import time
from typing import Dict, Any, Tuple, List, Callable, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """Configuration management class for pipeline settings."""
    
    def __init__(self, config_dict: Dict = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None):
        """Get a configuration value with dot notation support."""
        if "." in key:
            keys = key.split(".")
            current = self.config_dict
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            return current
        return self.config_dict.get(key, default)

    @staticmethod
    def load_file(config_path: str):
        """Loads configuration from a YAML or JSON file."""
        try:
            with open(config_path, "r") as file:
                if config_path.endswith((".yml", ".yaml")):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
            return Config(config_data)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")


class ArtifactStore:
    """Stores and retrieves intermediate artifacts for the pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.base_path = config.get("folder_path.artifacts", "artifacts")
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Artifact store initialized at '{self.base_path}'")

    def save_artifact(
        self,
        artifact: Any,
        subdir: str,
        name: str,
        run_id: str = None
    ) -> str:
        """Save an artifact in the specified format and return the path."""
        # Include run_id in path if provided
        if run_id:
            artifact_dir = os.path.join(self.base_path, run_id, subdir)
        else:
            artifact_dir = os.path.join(self.base_path, subdir)
            
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, name)

        if name.endswith(".pkl"):
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact, f)
        elif name.endswith(".csv"):
            if isinstance(artifact, pd.DataFrame):
                artifact.to_csv(artifact_path, index=False)
            else:
                raise ValueError("CSV format only supports pandas DataFrames.")
        elif name.endswith((".json")):
            with open(artifact_path, "w") as f:
                json.dump(artifact, f, indent=2)
        else:
            raise ValueError(f"Unsupported format for {name}")
            
        logging.info(f"Artifact '{name}' saved to {artifact_path}")
        return artifact_path

    def load_artifact(
        self,
        subdir: str,
        name: str,
        run_id: str = None
    ):
        """Load an artifact in the specified format."""
        if run_id:
            artifact_path = os.path.join(self.base_path, run_id, subdir, name)
        else:
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
            logging.warning(f"Artifact '{name}' not found in {artifact_path}")
            return None
            
    def list_artifacts(self, run_id=None):
        """List all artifacts or artifacts for a specific run."""
        artifacts = []
        
        if run_id:
            base_dir = os.path.join(self.base_path, run_id)
            if not os.path.exists(base_dir):
                return []
        else:
            base_dir = self.base_path
            
        for root, _, files in os.walk(base_dir):
            for file in files:
                artifact_path = os.path.join(root, file)
                artifacts.append(artifact_path)
                
        return artifacts


def node(name: str = None, stage: int = None, dependencies: List[str] = None):
    """Decorator to mark functions as pipeline nodes."""
    def decorator(func: Callable) -> Callable:
        # Add metadata to the function
        func._is_node = True  # Mark as a node for discovery
        func._node_metadata = {
            "name": name or func.__name__,
            "stage": stage or 0,
            "dependencies": dependencies or []
        }
        return func
    return decorator


class Stack:
    """A stack that brings together all components needed to run pipelines."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = Config(config or {})
        self.artifact_store = None
        self.run_details = {}
        self.context = {}  # Store results from executed nodes
        
    def set_artifact_store(self, artifact_store: ArtifactStore):
        """Set the artifact store for this stack."""
        self.artifact_store = artifact_store
        return self
    
    def _discover_nodes(self, component):
        """Discover all node functions in a component."""
        nodes = []
        for name, method in inspect.getmembers(component, inspect.ismethod):
            if hasattr(method, '_is_node') and method._is_node:
                nodes.append(method)
        return nodes
        
    def run(self, components: List[Any]) -> str:
        """Discover nodes from components and execute them in order of their stage."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        logging.info(f"Starting pipeline run: {run_id}")
        
        # Discover nodes from all components
        all_nodes = []
        for component in components:
            component_nodes = self._discover_nodes(component)
            all_nodes.extend(component_nodes)
            
        # Sort nodes by stage
        sorted_nodes = sorted(all_nodes, key=lambda n: n._node_metadata["stage"])
        
        # Store run details
        self.run_details[run_id] = {
            "pipeline_name": self.name,
            "status": "running",
            "start_time": start_time,
            "tasks": []
        }
        
        try:
            # Execute nodes in order
            for node_func in sorted_nodes:
                metadata = node_func._node_metadata
                node_name = metadata["name"]
                node_stage = metadata["stage"]
                node_dependencies = metadata["dependencies"]
                
                logging.info(f"Running node '{node_name}' at stage {node_stage}")
                
                # Check if dependencies are satisfied
                missing_deps = [dep for dep in node_dependencies if dep not in self.context]
                if missing_deps:
                    raise ValueError(f"Node '{node_name}' has unsatisfied dependencies: {missing_deps}")
                
                # Prepare arguments for the node
                node_kwargs = {
                    "config": self.config,
                    "artifact_store": self.artifact_store,
                    "run_id": run_id
                }
                
                # Add dependencies to node arguments
                for dep in node_dependencies:
                    node_kwargs[dep] = self.context[dep]
                
                # Execute the node
                node_start_time = time.time()
                result = node_func(**node_kwargs)
                node_end_time = time.time()
                
                # Store result in context
                self.context[node_name] = result
                
                # Update run details
                self.run_details[run_id]["tasks"].append({
                    "name": node_name,
                    "stage": node_stage,
                    "duration_seconds": node_end_time - node_start_time
                })
            
            # Mark run as completed
            end_time = time.time()
            self.run_details[run_id].update({
                "status": "completed",
                "end_time": end_time,
                "duration_seconds": end_time - start_time
            })
            
            logging.info(f"Pipeline run {run_id} completed successfully")
            
        except Exception as e:
            # Mark run as failed
            end_time = time.time()
            self.run_details[run_id].update({
                "status": "failed",
                "end_time": end_time,
                "duration_seconds": end_time - start_time,
                "error": str(e)
            })
            logging.error(f"Pipeline run {run_id} failed: {e}")
            raise
            
        return run_id
        
    def get_run_details(self, run_id: str) -> Dict:
        """Get details about a specific run."""
        return self.run_details.get(run_id, {"status": "unknown", "error": "Run not found"})


class DataIngestion:
    """Handle data ingestion operations."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    @node(name="data_loader", stage=1)
    def ingest_data(self, config: Config, artifact_store: ArtifactStore, 
                   run_id: str) -> Dict[str, pd.DataFrame]:
        """Load and split data into train and test sets."""
        # Define paths for artifacts
        raw_path = config.get("folder_path.raw_data", "raw_data")
        raw_train_filename = config.get("filenames.raw_train", "train_data.csv")
        raw_test_filename = config.get("filenames.raw_test", "test_data.csv")
        
        # Load raw data
        logging.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Split data
        test_size = config.get("base.test_size", 0.2)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        logging.info(
            f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}"
        )
        
        # Save raw artifacts
        artifact_store.save_artifact(
            train_data, subdir=raw_path, name=raw_train_filename, run_id=run_id
        )
        artifact_store.save_artifact(
            test_data, subdir=raw_path, name=raw_test_filename, run_id=run_id
        )
        
        logging.info("Data ingestion completed")
        return {
            "train_data": train_data,
            "test_data": test_data
        }


class DataProcessor:
    """Handle data processing operations."""
    
    @node(name="data_processor", stage=2, dependencies=["data_loader"])
    def process_data(self, data_loader: Dict[str, pd.DataFrame], config: Config, 
                    artifact_store: ArtifactStore, run_id: str) -> Dict[str, pd.DataFrame]:
        """Process the train and test data."""
        # Extract data from previous node
        train_data = data_loader["train_data"]
        test_data = data_loader["test_data"]
        
        # Define paths for artifacts
        processed_path = config.get("folder_path.processed_data", "processed_data")
        processed_train_filename = config.get("filenames.processed_train", "processed_train.csv")
        processed_test_filename = config.get("filenames.processed_test", "processed_test.csv")
        
        # Implement your data processing logic here
        # This is a placeholder - add your actual processing steps
        processed_train = train_data.copy()
        processed_test = test_data.copy()
        
        # Example processing: fill missing values
        for df in [processed_train, processed_test]:
            for col in df.select_dtypes(include=['number']).columns:
                df[col] = df[col].fillna(df[col].mean())
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")
        
        # Save processed artifacts
        artifact_store.save_artifact(
            processed_train, subdir=processed_path, name=processed_train_filename, run_id=run_id
        )
        artifact_store.save_artifact(
            processed_test, subdir=processed_path, name=processed_test_filename, run_id=run_id
        )
        
        logging.info("Data processing completed")
        return {
            "processed_train": processed_train,
            "processed_test": processed_test
        }


class ModelTrainer:
    """Handle model training operations."""
    
    @node(name="model_trainer", stage=3, dependencies=["data_processor"])
    def train_model(self, data_processor: Dict[str, pd.DataFrame], config: Config, 
                   artifact_store: ArtifactStore, run_id: str) -> Dict[str, Any]:
        """Train a model on the processed data."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Extract data from previous node
        processed_train = data_processor["processed_train"]
        
        # Define paths for artifacts
        models_path = config.get("folder_path.models", "models")
        model_filename = config.get("filenames.model", "model.pkl")
        
        # Prepare data
        target_col = config.get("base.target_column", "target")
        feature_cols = [col for col in processed_train.columns if col != target_col]
        
        X_train = processed_train[feature_cols]
        y_train = processed_train[target_col]
        
        # Train model
        model_type = config.get("model.type", "random_forest")
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=config.get("model.params.n_estimators", 100),
                max_depth=config.get("model.params.max_depth", None),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logging.info(f"Training {model_type} model")
        model.fit(X_train, y_train)
        
        # Save model
        artifact_store.save_artifact(
            model, subdir=models_path, name=model_filename, run_id=run_id
        )
        
        # Save feature importance
        if hasattr(model, "feature_importances_"):
            feature_importance = {
                feature: importance 
                for feature, importance in zip(feature_cols, model.feature_importances_)
            }
            artifact_store.save_artifact(
                feature_importance, 
                subdir=models_path, 
                name="feature_importance.json", 
                run_id=run_id
            )
        
        logging.info("Model training completed")
        return {
            "model": model,
            "feature_cols": feature_cols,
            "target_col": target_col
        }


class ModelEvaluator:
    """Handle model evaluation operations."""
    
    @node(name="model_evaluator", stage=4, dependencies=["model_trainer", "data_processor"])
    def evaluate_model(self, model_trainer: Dict[str, Any], data_processor: Dict[str, pd.DataFrame], 
                      config: Config, artifact_store: ArtifactStore, run_id: str) -> Dict[str, Any]:
        """Evaluate the trained model on the test data."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Extract data and model from previous nodes
        processed_test = data_processor["processed_test"]
        model = model_trainer["model"]
        feature_cols = model_trainer["feature_cols"]
        target_col = model_trainer["target_col"]
        
        # Define paths for artifacts
        metrics_path = config.get("folder_path.metrics", "metrics")
        metrics_filename = config.get("filenames.metrics", "metrics.json")
        
        # Prepare test data
        X_test = processed_test[feature_cols]
        y_test = processed_test[target_col]
        
        # Make predictions
        logging.info("Evaluating model on test data")
        y_pred = model.predict(X_test)
        
        # For probability-based metrics
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
        
        # Add ROC AUC if applicable (binary classification with probabilities)
        if y_pred_proba is not None and len(set(y_test)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
        
        # Save metrics
        artifact_store.save_artifact(
            metrics, subdir=metrics_path, name=metrics_filename, run_id=run_id
        )
        
        logging.info(f"Model evaluation completed with accuracy: {metrics['accuracy']:.4f}")
        return metrics


class TrainingPipeline:
    """Main pipeline class that orchestrates the training workflow."""
    
    def __init__(self, data_path: str, config_path: str):
        self.data_path = data_path
        # Load configuration
        self.config = Config.load_file(config_path)
        self.artifact_store = ArtifactStore(self.config)
        self.stack = Stack("Training Pipeline", self.config.config_dict)
        self.stack.set_artifact_store(self.artifact_store)
        
        # Initialize components
        self.data_ingestion = DataIngestion(data_path)
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        
        # Components to use in the pipeline
        self.components = [
            self.data_ingestion,
            self.data_processor,
            self.model_trainer,
            self.model_evaluator
        ]
    
    def run(self):
        """Run the complete pipeline."""
        try:
            # Execute pipeline with auto-discovered nodes
            run_id = self.stack.run(self.components)
            
            # List artifacts
            artifacts = self.artifact_store.list_artifacts(run_id)
            print(f"\nRun ID: {run_id}")
            print("Artifacts:")
            for uri in artifacts:
                print(f"- {uri}")

            run_details = self.stack.get_run_details(run_id)
            print("\nRun Details:")
            print(f"Pipeline: {run_details.get('pipeline_name')}")
            print(f"Status: {run_details.get('status')}")
            print(f"Duration: {run_details.get('duration_seconds', 0):.2f} seconds")
            print(f"Tasks executed: {len(run_details.get('tasks', []))}")
            
            # Check if run was successful
            if run_details.get("status") == "completed":
                print("Pipeline completed successfully")
            else:
                print(f"Pipeline failed: {run_details.get('error', 'Unknown error')}")
            
            return run_id
            
        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            raise


def sample_config():
    """Generate a sample configuration file."""
    config = {
        "folder_path": {
            "artifacts": "artifacts",
            "raw_data": "raw_data",
            "processed_data": "processed_data",
            "models": "models",
            "metrics": "metrics"
        },
        "filenames": {
            "raw_train": "train_data.csv",
            "raw_test": "test_data.csv",
            "processed_train": "processed_train.csv",
            "processed_test": "processed_test.csv",
            "model": "model.pkl",
            "metrics": "metrics.json"
        },
        "base": {
            "test_size": 0.2,
            "target_column": "target"
        },
        "model": {
            "type": "random_forest",
            "params": {
                "n_estimators": 100,
                "max_depth": 10
            }
        }
    }
    
    os.makedirs("config", exist_ok=True)
    with open("config/sample_config.yml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Sample configuration written to config/sample_config.yml")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the training pipeline")
    parser.add_argument("--data", required=True, help="Path to the input data CSV file")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--generate-config", action="store_true", help="Generate a sample configuration file")
    
    args = parser.parse_args()
    
    if args.generate_config:
        os.makedirs("config", exist_ok=True)
        sample_config()
    
    if args.data and args.config:
        pipeline = TrainingPipeline(args.data, args.config)
        run_id = pipeline.run()
        logging.info(f"Pipeline executed successfully with run ID: {run_id}")





# Create a pipeline
pipeline = TrainingPipeline("data.csv", "config.yml")

# Run pipeline - automatically discovers and executes nodes
run_id = pipeline.run()