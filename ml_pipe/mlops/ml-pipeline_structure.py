import os
import logging
import json
import yaml
import pickle
from typing import Dict, Any, Tuple, List, Callable
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    def __init__(self, config_dict: Dict = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None):
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

    def __init__(self, config):
        self.config = config
        self.base_path = self.config.get("folder_path", {}).get(
            "artifacts", "artifacts"
        )
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Artifact store initialized at '{self.base_path}'")

    def save_artifact(
        self,
        artifact: Any,
        subdir: str,
        name: str,
    ) -> str:
        """Save an artifact in the specified format and return the path."""
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
        else:
            raise ValueError(f"Unsupported format for {name}")
        logging.info(f"Artifact '{name}' saved to {artifact_path}")
        return artifact_path

    def load_artifact(
        self,
        subdir: str,
        name: str,
    ):
        """Load an artifact in the specified format."""
        artifact_path = os.path.join(self.base_path, subdir, name)
        if os.path.exists(artifact_path):
            if name.endswith(".pkl"):
                with open(artifact_path, "rb") as f:
                    artifact = pickle.load(f)
            elif name.endswith(".csv"):
                artifact = pd.read_csv(artifact_path)
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
        for root, _, files in os.walk(self.base_path):
            for file in files:
                artifact_path = os.path.join(root, file)
                # If run_id is specified, only include artifacts containing that run_id
                if run_id is None or run_id in artifact_path:
                    artifacts.append(artifact_path)
        return artifacts


class PipelineNode:
    """Represents a single task/node in the pipeline."""
    
    def __init__(self, func: Callable, name: str, task_id: int):
        self.func = func
        self.name = name
        self.task_id = task_id
        
    def execute(self, *args, **kwargs):
        logging.info(f"Executing task {self.task_id}: {self.name}")
        return self.func(*args, **kwargs)


class Pipeline:
    """Pipeline class to orchestrate the execution of nodes."""
    
    def __init__(self, name: str):
        self.name = name
        self.nodes = []
        
    def add_node(self, node: PipelineNode):
        """Add a node to the pipeline."""
        self.nodes.append(node)
        self.nodes.sort(key=lambda x: x.task_id)  # Sort nodes by task_id
        
    def run(self, *args, **kwargs):
        """Execute the pipeline nodes in order."""
        logging.info(f"Starting pipeline: {self.name}")
        results = {}
        
        # Pass initial arguments to the first node
        if self.nodes:
            results[self.nodes[0].name] = self.nodes[0].execute(*args, **kwargs)
            
            # Execute subsequent nodes with results from previous nodes
            for i in range(1, len(self.nodes)):
                current_node = self.nodes[i]
                prev_results = list(results.values())
                results[current_node.name] = current_node.execute(*prev_results, **kwargs)
                
        logging.info(f"Pipeline {self.name} completed")
        return results


# Decorator to create task functions
def node(name: str = None, task_id: int = None):
    """Decorator to mark functions as pipeline nodes."""
    def decorator(func):
        func._node_metadata = {
            "name": name or func.__name__,
            "task_id": task_id or 0
        }
        return func
    return decorator


class DataIngestion:
    def __init__(self, config, artifact_store):
        self.config = config
        self.artifact_store = artifact_store

    @node(name="data_loader", task_id=1)
    def data_ingestion(self, path):
        """Load and split data into train and test sets."""
        # Define paths for artifacts
        raw_path = self.config.get("folder_path", {}).get("raw_data", "raw_data")
        raw_train_filename = self.config.get("filenames", {}).get("raw_train", "train_data.csv")
        raw_test_filename = self.config.get("filenames", {}).get("raw_test", "test_data.csv")
        
        # Load raw data
        df = pd.read_csv(path)
        # Split data
        test_size = self.config.get("base", {}).get("test_size", 0.2)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        logging.info(
            f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}"
        )
        # Save raw artifacts
        self.artifact_store.save_artifact(
            train_data, subdir=raw_path, name=raw_train_filename 
        )
        self.artifact_store.save_artifact(
            test_data, subdir=raw_path, name=raw_test_filename
        )
        logging.info("Data ingestion completed")
        return train_data, test_data


class DataProcessor:
    def __init__(self, config, artifact_store):
        self.config = config
        self.artifact_store = artifact_store

    @node(name="data_processor", task_id=2)
    def process_data(self, data_tuple):
        """Process the train and test data."""
        train_data, test_data = data_tuple
        
        # Define paths for artifacts
        processed_path = self.config.get("folder_path", {}).get("processed_data", "processed_data")
        processed_train_filename = self.config.get("filenames", {}).get("processed_train", "processed_train.csv")
        processed_test_filename = self.config.get("filenames", {}).get("processed_test", "processed_test.csv")
        
        # Simple processing example (replace with your actual processing logic)
        train_processed = self._process(train_data)
        test_processed = self._process(test_data)
        
        # Save processed artifacts
        self.artifact_store.save_artifact(
            train_processed, subdir=processed_path, name=processed_train_filename
        )
        self.artifact_store.save_artifact(
            test_processed, subdir=processed_path, name=processed_test_filename
        )
        
        logging.info("Data processing completed")
        return train_processed, test_processed
    
    def _process(self, data):
        """Internal method to process the data."""
        # Add your data processing logic here
        # Example: handle missing values, feature engineering, etc.
        processed_data = data.copy()
        
        # Example: Fill missing values
        processed_data = processed_data.fillna(0)
        
        # Example: Feature engineering (add a new column)
        if 'feature1' in processed_data.columns and 'feature2' in processed_data.columns:
            processed_data['feature_combined'] = processed_data['feature1'] * processed_data['feature2']
            
        return processed_data


class FeatureEngineer:
    def __init__(self, config, artifact_store):
        self.config = config
        self.artifact_store = artifact_store

    @node(name="feature_engineer", task_id=3)
    def engineer_features(self, data_tuple):
        """Engineer features from the processed data."""
        train_processed, test_processed = data_tuple
        
        # Define paths for artifacts
        features_path = self.config.get("folder_path", {}).get("features", "features")
        features_train_filename = self.config.get("filenames", {}).get("features_train", "features_train.csv")
        features_test_filename = self.config.get("filenames", {}).get("features_test", "features_test.csv")
        
        # Feature engineering logic
        train_features = self._engineer_features(train_processed)
        test_features = self._engineer_features(test_processed)
        
        # Save feature artifacts
        self.artifact_store.save_artifact(
            train_features, subdir=features_path, name=features_train_filename
        )
        self.artifact_store.save_artifact(
            test_features, subdir=features_path, name=features_test_filename
        )
        
        logging.info("Feature engineering completed")
        return train_features, test_features
    
    def _engineer_features(self, data):
        """Internal method to engineer features."""
        # Add your feature engineering logic here
        features = data.copy()
        
        # Example: Scaling numerical features
        numerical_cols = features.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            if col in features.columns:
                features[f"{col}_scaled"] = (features[col] - features[col].mean()) / features[col].std()
        
        return features


class ModelTrainer:
    def __init__(self, config, artifact_store):
        self.config = config
        self.artifact_store = artifact_store

    @node(name="model_trainer", task_id=4)
    def train_model(self, data_tuple):
        """Train a model using the engineered features."""
        train_features, test_features = data_tuple
        
        # Define paths for artifacts
        models_path = self.config.get("folder_path", {}).get("models", "models")
        model_filename = self.config.get("filenames", {}).get("model", "model.pkl")
        
        # Separate features and target
        target_col = self.config.get("model", {}).get("target_column", "target")
        
        X_train = train_features.drop(columns=[target_col], errors='ignore')
        y_train = train_features[target_col] if target_col in train_features.columns else None
        
        X_test = test_features.drop(columns=[target_col], errors='ignore')
        y_test = test_features[target_col] if target_col in test_features.columns else None
        
        # Train model
        model = self._train_model(X_train, y_train)
        
        # Save model artifact
        self.artifact_store.save_artifact(
            model, subdir=models_path, name=model_filename
        )
        
        logging.info("Model training completed")
        return model, X_test, y_test
    
    def _train_model(self, X, y):
        """Internal method to train a model."""
        # Add your model training logic here
        # Example: train a simple model
        from sklearn.ensemble import RandomForestClassifier
        
        model_type = self.config.get("model", {}).get("type", "random_forest")
        
        if model_type == "random_forest":
            n_estimators = self.config.get("model", {}).get("n_estimators", 100)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        else:
            # Default to RandomForest if model type not recognized
            model = RandomForestClassifier(random_state=42)
        
        model.fit(X, y)
        return model


class ModelEvaluator:
    def __init__(self, config, artifact_store):
        self.config = config
        self.artifact_store = artifact_store

    @node(name="model_evaluator", task_id=5)
    def evaluate_model(self, model_data):
        """Evaluate the trained model."""
        model, X_test, y_test = model_data
        
        # Define paths for artifacts
        metrics_path = self.config.get("folder_path", {}).get("metrics", "metrics")
        metrics_filename = self.config.get("filenames", {}).get("metrics", "metrics.json")
        
        # Evaluate model
        metrics = self._evaluate_model(model, X_test, y_test)
        
        # Save metrics artifact
        self.artifact_store.save_artifact(
            metrics, subdir=metrics_path, name=metrics_filename
        )
        
        logging.info("Model evaluation completed")
        return metrics
    
    def _evaluate_model(self, model, X, y):
        """Internal method to evaluate a model."""
        # Add your model evaluation logic here
        # Example: calculate common metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = model.predict(X)
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='weighted'),
            "recall": recall_score(y, y_pred, average='weighted'),
            "f1": f1_score(y, y_pred, average='weighted')
        }
        
        logging.info(f"Model metrics: {metrics}")
        return metrics


class MLPipeline:
    """Main ML pipeline class that orchestrates the entire workflow."""

    def __init__(self, data_path: str, config_path: str):
        # Load configuration
        self.config = Config.load_file(config_path)
        self.data_path = data_path
        
        # Initialize artifact store
        self.artifact_store = ArtifactStore(self.config)
        
        # Initialize pipeline components
        self.data_ingestion = DataIngestion(self.config, self.artifact_store)
        self.data_processor = DataProcessor(self.config, self.artifact_store)
        self.feature_engineer = FeatureEngineer(self.config, self.artifact_store)
        self.model_trainer = ModelTrainer(self.config, self.artifact_store)
        self.model_evaluator = ModelEvaluator(self.config, self.artifact_store)
        
        # Create pipeline
        self.pipeline = Pipeline("ml_training_pipeline")
        
        # Create nodes from component methods
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Set up the pipeline with nodes for each component."""
        # Create nodes from component methods with metadata
        data_ingestion_node = PipelineNode(
            func=self.data_ingestion.data_ingestion,
            name=self.data_ingestion.data_ingestion._node_metadata["name"],
            task_id=self.data_ingestion.data_ingestion._node_metadata["task_id"]
        )
        
        data_processor_node = PipelineNode(
            func=self.data_processor.process_data,
            name=self.data_processor.process_data._node_metadata["name"],
            task_id=self.data_processor.process_data._node_metadata["task_id"]
        )
        
        feature_engineer_node = PipelineNode(
            func=self.feature_engineer.engineer_features,
            name=self.feature_engineer.engineer_features._node_metadata["name"],
            task_id=self.feature_engineer.engineer_features._node_metadata["task_id"]
        )
        
        model_trainer_node = PipelineNode(
            func=self.model_trainer.train_model,
            name=self.model_trainer.train_model._node_metadata["name"],
            task_id=self.model_trainer.train_model._node_metadata["task_id"]
        )
        
        model_evaluator_node = PipelineNode(
            func=self.model_evaluator.evaluate_model,
            name=self.model_evaluator.evaluate_model._node_metadata["name"],
            task_id=self.model_evaluator.evaluate_model._node_metadata["task_id"]
        )
        
        # Add nodes to pipeline
        self.pipeline.add_node(data_ingestion_node)
        self.pipeline.add_node(data_processor_node)
        self.pipeline.add_node(feature_engineer_node)
        self.pipeline.add_node(model_trainer_node)
        self.pipeline.add_node(model_evaluator_node)

    def run(self):
        """Execute the full ML pipeline."""
        logging.info(f"Starting ML pipeline with data: {self.data_path}")
        results = self.pipeline.run(self.data_path)
        logging.info("ML pipeline completed successfully")
        return results


# Example configuration file structure (to be saved as config/config.yml)
def create_example_config():
    """Create an example configuration file."""
    config = {
        "base": {
            "test_size": 0.2,
            "random_state": 42
        },
        "folder_path": {
            "artifacts": "artifacts",
            "raw_data": "raw_data",
            "processed_data": "processed_data",
            "features": "features",
            "models": "models",
            "metrics": "metrics"
        },
        "filenames": {
            "raw_train": "train_data.csv",
            "raw_test": "test_data.csv",
            "processed_train": "processed_train.csv",
            "processed_test": "processed_test.csv",
            "features_train": "features_train.csv",
            "features_test": "features_test.csv",
            "model": "model.pkl",
            "metrics": "metrics.json"
        },
        "model": {
            "type": "random_forest",
            "n_estimators": 100,
            "max_depth": 10,
            "target_column": "target"
        }
    }
    
    os.makedirs("config", exist_ok=True)
    with open("config/config.yml", "w") as f:
        yaml.dump(config, f)
    
    logging.info("Example configuration file created at config/config.yml")


if __name__ == "__main__":
    # Create example config if it doesn't exist
    if not os.path.exists("config/config.yml"):
        create_example_config()
    
    data_path = "data.csv"
    config_path = "config/config.yml"
    
    # Run the ML pipeline
    pipeline = MLPipeline(data_path, config_path)
    results = pipeline.run()
    
    # Print metrics
    if "model_evaluator" in results:
        metrics = results["model_evaluator"]
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
