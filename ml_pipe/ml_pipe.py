import os
import logging
import json
import yaml
import pickle
from typing import Dict, Any, List, Callable, Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    def __init__(self, config: Config):
        self.config = config
        self.base_path = self.config.get("folder_path", {}).get("artifacts", "artifacts")
        os.makedirs(self.base_path, exist_ok=True)
        logger.info(f"Artifact store initialized at '{self.base_path}'")

    def save_artifact(self, artifact: Any, subdir: str, name: str) -> str:
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
        elif name.endswith((".json")):
            with open(artifact_path, "w") as f:
                json.dump(artifact, f, indent=4)
        else:
            raise ValueError(f"Unsupported format for {name}")
            
        logger.info(f"Artifact '{name}' saved to {artifact_path}")
        return artifact_path

    def load_artifact(self, subdir: str, name: str) -> Any:
        """Load an artifact in the specified format."""
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
                
            logger.info(f"Artifact '{name}' loaded from {artifact_path}")
            return artifact
        else:
            logger.warning(f"Artifact '{name}' not found in {artifact_path}")
            return None

    def list_artifacts(self, run_id: Optional[str] = None) -> List[str]:
        """List all artifacts or artifacts for a specific run."""
        artifacts = []
        for root, _, files in os.walk(self.base_path):
            for file in files:
                artifact_path = os.path.join(root, file)
                # If run_id is specified, only include artifacts containing that run_id
                if run_id is None or run_id in artifact_path:
                    artifacts.append(artifact_path)
        return artifacts


class Node:
    """Class representing a pipeline node with its metadata and function"""
    def __init__(self, func: Callable, name: str = None, stage: int = None, dependencies: List[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.stage = stage
        self.dependencies = dependencies or []
        
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class Pipeline:
    """Pipeline orchestrator to manage the execution of nodes"""
    def __init__(self, config: Config, artifact_store: ArtifactStore):
        self.config = config
        self.artifact_store = artifact_store
        self.nodes: Dict[str, Node] = {}
        self.results: Dict[str, Any] = {}
        self.run_id = None
        
    def add_node(self, node: Node) -> None:
        """Add a node to the pipeline"""
        self.nodes[node.name] = node
        logger.info(f"Added node '{node.name}' to pipeline (stage: {node.stage})")
        
    def generate_run_id(self) -> str:
        """Generate a unique run ID for the pipeline execution"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run(self) -> Dict[str, Any]:
        """Execute the pipeline in the correct order based on node dependencies and stages"""
        self.run_id = self.generate_run_id()
        logger.info(f"Starting pipeline run with ID: {self.run_id}")
        
        # Sort nodes by stage
        sorted_nodes = sorted(self.nodes.values(), key=lambda x: x.stage or float('inf'))
        
        # Execute nodes in order
        for node in sorted_nodes:
            logger.info(f"Executing node '{node.name}' (stage: {node.stage})")
            
            # Prepare arguments for the node
            args = []
            for dep in node.dependencies:
                if dep not in self.results:
                    raise ValueError(f"Dependency '{dep}' for node '{node.name}' not found in results")
                args.append(self.results[dep])
                
            # Execute the node
            result = node(self.config, self.artifact_store, *args)
            self.results[node.name] = result
            
            logger.info(f"Node '{node.name}' completed successfully")
            
        logger.info(f"Pipeline run {self.run_id} completed successfully")
        return self.results


# Node decorator
def pipeline_node(name: str = None, stage: int = None, dependencies: List[str] = None):
    """Decorator to mark functions as pipeline nodes"""
    def decorator(func):
        node = Node(
            func=func,
            name=name or func.__name__,
            stage=stage,
            dependencies=dependencies or []
        )
        return node
    return decorator


# ===== Pipeline Nodes =====

@pipeline_node(name="data_loader", stage=1)
def data_ingestion(config: Config, artifact_store: ArtifactStore, data_path: str):
    """Load and split the raw dataset"""
    # Define paths for artifacts
    raw_path = config.get("folder_path", {}).get("raw_data", "raw_data")
    raw_train_filename = config.get("filenames", {}).get("raw_train", "train_data.csv")
    raw_test_filename = config.get("filenames", {}).get("raw_test", "test_data.csv")

    # Load raw data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Split data
    test_size = config.get("base", {}).get("test_size", 0.2)
    train_data, test_data = train_test_split(
        df, 
        test_size=test_size, 
        random_state=config.get("base", {}).get("random_state", 42)
    )
    logger.info(f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    # Save raw artifacts
    artifact_store.save_artifact(train_data, subdir=raw_path, name=raw_train_filename)
    artifact_store.save_artifact(test_data, subdir=raw_path, name=raw_test_filename)
    
    return {
        "train_data": train_data,
        "test_data": test_data
    }


@pipeline_node(name="data_processor", stage=2, dependencies=["data_loader"])
def data_processing(config: Config, artifact_store: ArtifactStore, data_dict: Dict):
    """Process the raw dataset"""
    train_data = data_dict["train_data"]
    test_data = data_dict["test_data"]
    
    # Define paths for artifacts
    process_path = config.get("folder_path", {}).get("processed_data", "processed_data")
    process_train_filename = config.get("filenames", {}).get("processed_train", "processed_train_data.csv")
    process_test_filename = config.get("filenames", {}).get("processed_test", "processed_test_data.csv")

    # Process data
    logger.info("Processing training and test data")
    
    # Example processing steps (replace with your actual processing logic)
    train_processed = process_dataset(train_data, config)
    test_processed = process_dataset(test_data, config)

    # Save artifacts
    artifact_store.save_artifact(train_processed, subdir=process_path, name=process_train_filename)
    artifact_store.save_artifact(test_processed, subdir=process_path, name=process_test_filename)
    
    return {
        "train_processed": train_processed,
        "test_processed": test_processed
    }


@pipeline_node(name="feature_engineer", stage=3, dependencies=["data_processor"])
def feature_engineering(config: Config, artifact_store: ArtifactStore, processed_data: Dict):
    """Extract features from processed data"""
    train_processed = processed_data["train_processed"]
    test_processed = processed_data["test_processed"]
    
    # Define paths for artifacts
    feature_path = config.get("folder_path", {}).get("features", "features")
    feature_train_filename = config.get("filenames", {}).get("feature_train", "feature_train_data.csv")
    feature_test_filename = config.get("filenames", {}).get("feature_test", "feature_test_data.csv")

    # Feature engineering
    logger.info("Performing feature engineering")
    
    # Example feature engineering (replace with your actual feature engineering logic)
    train_features = engineer_features(train_processed, config)
    test_features = engineer_features(test_processed, config)

    # Save artifacts
    artifact_store.save_artifact(train_features, subdir=feature_path, name=feature_train_filename)
    artifact_store.save_artifact(test_features, subdir=feature_path, name=feature_test_filename)
    
    return {
        "train_features": train_features,
        "test_features": test_features
    }


@pipeline_node(name="model_trainer", stage=4, dependencies=["feature_engineer"])
def model_training(config: Config, artifact_store: ArtifactStore, features: Dict):
    """Train machine learning models"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    train_features = features["train_features"]
    
    # Define paths for artifacts
    model_path = config.get("folder_path", {}).get("models", "models")
    
    # Prepare data
    feature_cols = config.get("model", {}).get("feature_columns", [])
    if not feature_cols:  # If not specified, use all columns except target
        target_col = config.get("model", {}).get("target_column", "target")
        feature_cols = [col for col in train_features.columns if col != target_col]
    
    target_col = config.get("model", {}).get("target_column", "target")
    
    X_train = train_features[feature_cols]
    y_train = train_features[target_col]
    
    # Train multiple models
    logger.info("Training machine learning models")
    models = {}
    
    # Model 1: Random Forest
    rf_params = config.get("model", {}).get("random_forest", {})
    rf = RandomForestClassifier(
        n_estimators=rf_params.get("n_estimators", 100),
        max_depth=rf_params.get("max_depth", None),
        random_state=config.get("base", {}).get("random_state", 42)
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf
    
    # Model 2: Gradient Boosting
    gb_params = config.get("model", {}).get("gradient_boosting", {})
    gb = GradientBoostingClassifier(
        n_estimators=gb_params.get("n_estimators", 100),
        learning_rate=gb_params.get("learning_rate", 0.1),
        random_state=config.get("base", {}).get("random_state", 42)
    )
    gb.fit(X_train, y_train)
    models["gradient_boosting"] = gb
    
    # Model 3: Logistic Regression
    lr_params = config.get("model", {}).get("logistic_regression", {})
    lr = LogisticRegression(
        C=lr_params.get("C", 1.0),
        max_iter=lr_params.get("max_iter", 100),
        random_state=config.get("base", {}).get("random_state", 42)
    )
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr
    
    # Save models
    for model_name, model in models.items():
        model_filename = f"{model_name}_model.pkl"
        artifact_store.save_artifact(model, subdir=model_path, name=model_filename)
    
    return {
        "models": models,
        "feature_columns": feature_cols,
        "target_column": target_col
    }


@pipeline_node(name="model_evaluator", stage=5, dependencies=["model_trainer", "feature_engineer"])
def model_evaluation(config: Config, artifact_store: ArtifactStore, model_data: Dict, features: Dict):
    """Evaluate trained models on test data"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    models = model_data["models"]
    feature_cols = model_data["feature_columns"]
    target_col = model_data["target_column"]
    test_features = features["test_features"]
    
    # Define paths for artifacts
    eval_path = config.get("folder_path", {}).get("evaluation", "evaluation")
    
    # Prepare test data
    X_test = test_features[feature_cols]
    y_test = test_features[target_col]
    
    # Evaluate models
    logger.info("Evaluating models on test data")
    metrics = {}
    
    for model_name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        model_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
        
        # Add ROC AUC if probability predictions are available
        if y_prob is not None:
            model_metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        
        metrics[model_name] = model_metrics
        logger.info(f"Evaluation metrics for {model_name}: {model_metrics}")
    
    # Save evaluation results
    artifact_store.save_artifact(metrics, subdir=eval_path, name="model_evaluation.json")
    
    return {
        "metrics": metrics
    }


@pipeline_node(name="model_stacker", stage=6, dependencies=["model_trainer", "feature_engineer"])
def model_stacking(config: Config, artifact_store: ArtifactStore, model_data: Dict, features: Dict):
    """Stack multiple models to create an ensemble model"""
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    models = model_data["models"]
    feature_cols = model_data["feature_columns"]
    target_col = model_data["target_column"]
    train_features = features["train_features"]
    
    # Define paths for artifacts
    model_path = config.get("folder_path", {}).get("models", "models")
    
    # Prepare data
    X_train = train_features[feature_cols]
    y_train = train_features[target_col]
    
    # Create list of base estimators for stacking
    estimators = [(name, model) for name, model in models.items()]
    
    # Create and train stacked model
    logger.info("Creating stacked ensemble model")
    stacked_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
    stacked_model.fit(X_train, y_train)
    
    # Save stacked model
    artifact_store.save_artifact(stacked_model, subdir=model_path, name="stacked_model.pkl")
    
    return {
        "stacked_model": stacked_model
    }


@pipeline_node(name="stacked_model_evaluator", stage=7, dependencies=["model_stacker", "feature_engineer"])
def stacked_model_evaluation(config: Config, artifact_store: ArtifactStore, stacked_model_data: Dict, features: Dict):
    """Evaluate the stacked model on test data"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    stacked_model = stacked_model_data["stacked_model"]
    test_features = features["test_features"]
    
    # Get feature and target columns from config or model data
    feature_cols = config.get("model", {}).get("feature_columns", [])
    target_col = config.get("model", {}).get("target_column", "target")
    
    # Define paths for artifacts
    eval_path = config.get("folder_path", {}).get("evaluation", "evaluation")
    
    # Prepare test data
    X_test = test_features[feature_cols] if feature_cols else test_features.drop(target_col, axis=1)
    y_test = test_features[target_col]
    
    # Evaluate stacked model
    logger.info("Evaluating stacked model on test data")
    y_pred = stacked_model.predict(X_test)
    y_prob = stacked_model.predict_proba(X_test)[:, 1] if hasattr(stacked_model, "predict_proba") else None
    
    # Calculate metrics
    stacked_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted')
    }
    
    # Add ROC AUC if probability predictions are available
    if y_prob is not None:
        stacked_metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    
    logger.info(f"Evaluation metrics for stacked model: {stacked_metrics}")
    
    # Save evaluation results
    artifact_store.save_artifact(stacked_metrics, subdir=eval_path, name="stacked_model_evaluation.json")
    
    return {
        "stacked_metrics": stacked_metrics
    }


# ===== Helper Functions =====

def process_dataset(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Process dataset with basic cleaning and transformations"""
    processed = df.copy()
    
    # Get columns to process from config
    numeric_cols = config.get("processing", {}).get("numeric_columns", [])
    categorical_cols = config.get("processing", {}).get("categorical_columns", [])
    
    # Fill missing values
    for col in numeric_cols:
        if col in processed.columns:
            # Fill with mean or specified value
            fill_value = config.get("processing", {}).get("numeric_fill", "mean")
            if fill_value == "mean":
                processed[col] = processed[col].fillna(processed[col].mean())
            elif fill_value == "median":
                processed[col] = processed[col].fillna(processed[col].median())
            else:
                processed[col] = processed[col].fillna(fill_value)
    
    for col in categorical_cols:
        if col in processed.columns:
            # Fill with mode or specified value
            fill_value = config.get("processing", {}).get("categorical_fill", "mode")
            if fill_value == "mode":
                processed[col] = processed[col].fillna(processed[col].mode()[0])
            else:
                processed[col] = processed[col].fillna(fill_value)
    
    # Handle outliers if configured
    if config.get("processing", {}).get("handle_outliers", False):
        for col in numeric_cols:
            if col in processed.columns:
                # Simple z-score based outlier handling
                z_scores = (processed[col] - processed[col].mean()) / processed[col].std()
                threshold = config.get("processing", {}).get("outlier_threshold", 3)
                processed.loc[abs(z_scores) > threshold, col] = processed[col].median()
    
    return processed


def engineer_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Create new features based on existing ones"""
    engineered = df.copy()
    
    # Get columns to use for feature engineering
    numeric_cols = config.get("feature_engineering", {}).get("numeric_columns", [])
    categorical_cols = config.get("feature_engineering", {}).get("categorical_columns", [])
    
    # One-hot encode categorical features
    if config.get("feature_engineering", {}).get("one_hot_encode", True):
        for col in categorical_cols:
            if col in engineered.columns:
                # Get dummies and add to dataframe
                dummies = pd.get_dummies(engineered[col], prefix=col, drop_first=True)
                engineered = pd.concat([engineered, dummies], axis=1)
                # Drop original column if configured
                if config.get("feature_engineering", {}).get("drop_original", True):
                    engineered = engineered.drop(col, axis=1)
    
    # Create interaction features if configured
    if config.get("feature_engineering", {}).get("create_interactions", False):
        interact_cols = config.get("feature_engineering", {}).get("interaction_columns", [])
        for i, col1 in enumerate(interact_cols):
            for col2 in interact_cols[i+1:]:
                if col1 in engineered.columns and col2 in engineered.columns:
                    engineered[f"{col1}_{col2}_interaction"] = engineered[col1] * engineered[col2]
    
    # Create polynomial features if configured
    if config.get("feature_engineering", {}).get("create_polynomial", False):
        poly_cols = config.get("feature_engineering", {}).get("polynomial_columns", [])
        degree = config.get("feature_engineering", {}).get("polynomial_degree", 2)
        for col in poly_cols:
            if col in engineered.columns:
                for d in range(2, degree + 1):
                    engineered[f"{col}_pow{d}"] = engineered[col] ** d
    
    return engineered


# ===== Pipeline Execution =====

class MLPipeline:
    """Main class for executing the ML pipeline"""
    def __init__(self, data_path: str, config_path: str):
        self.data_path = data_path
        self.config = Config.load_file(config_path)
        self.artifact_store = ArtifactStore(self.config)
        self.pipeline = Pipeline(self.config, self.artifact_store)
        
    def setup(self):
        """Set up the pipeline with all nodes"""
        # Add all nodes to the pipeline
        self.pipeline.add_node(data_ingestion)
        self.pipeline.add_node(data_processing)
        self.pipeline.add_node(feature_engineering)
        self.pipeline.add_node(model_training)
        self.pipeline.add_node(model_evaluation)
        self.pipeline.add_node(model_stacking)
        self.pipeline.add_node(stacked_model_evaluation)
        
    def run(self):
        """Run the complete pipeline"""
        # Set up the pipeline
        self.setup()
        
        # Execute data ingestion first with data path
        self.pipeline.results["data_loader"] = data_ingestion(
            self.config, 
            self.artifact_store, 
            self.data_path
        )
        
        # Continue with the rest of the pipeline
        results = self.pipeline.run()
        
        # Return the final results
        return results


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the ML pipeline")
    parser.add_argument("--data", default="data.csv", help="Path to input data CSV file")
    parser.add_argument("--config", default="config/config.yml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Create and run the pipeline
    pipeline = MLPipeline(args.data, args.config)
    results = pipeline.run()
    
    # Print evaluation results
    if "stacked_model_evaluator" in results:
        stacked_metrics = results["stacked_model_evaluator"]["stacked_metrics"]
        print("\nStacked Model Evaluation Results:")
        for metric, value in stacked_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    if "model_evaluator" in results:
        individual_metrics = results["model_evaluator"]["metrics"]
        print("\nIndividual Model Evaluation Results:")
        for model_name, metrics in individual_metrics.items():
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")