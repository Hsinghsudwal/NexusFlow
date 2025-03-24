import os
import logging
import json
import yaml
import pickle
from typing import Dict, Any, Tuple, List, Callable, Optional, Union
import pandas as pd
from datetime import datetime
import uuid
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

    def __init__(self, config: Config):
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
        elif name.endswith((".json")):
            with open(artifact_path, "w") as f:
                json.dump(artifact, f)
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
        for root, _, files in os.walk(self.base_path):
            for file in files:
                artifact_path = os.path.join(root, file)
                # If run_id is specified, only include artifacts containing that run_id
                if run_id is None or run_id in artifact_path:
                    artifacts.append(artifact_path)
        return artifacts


class Step:
    """Base class for all pipeline steps."""
    
    def __init__(self, name: str):
        self.name = name
        self.inputs = {}
        self.outputs = {}
        
    def execute(self, *args, **kwargs):
        """Execute the step logic. Must be implemented by subclasses."""
        raise NotImplementedError("Step subclasses must implement execute method")


class DataIngestionStep(Step):
    def __init__(self, config: Config, artifact_store: ArtifactStore):
        super().__init__(name="data_ingestion")
        self.config = config
        self.artifact_store = artifact_store

    def execute(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute data ingestion step."""
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
        
        self.outputs = {
            "train_data": train_data,
            "test_data": test_data
        }
        
        return train_data, test_data


class DataPreprocessingStep(Step):
    def __init__(self, config: Config, artifact_store: ArtifactStore):
        super().__init__(name="data_preprocessing")
        self.config = config
        self.artifact_store = artifact_store
        
    def execute(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute data preprocessing step."""
        self.inputs = {
            "train_data": train_data,
            "test_data": test_data
        }
        
        # Get preprocessing config
        preprocessing_config = self.config.get("preprocessing", {})
        
        # Process train data
        processed_train = self._preprocess_data(train_data, preprocessing_config)
        
        # Process test data
        processed_test = self._preprocess_data(test_data, preprocessing_config)
        
        # Save processed artifacts
        processed_path = self.config.get("folder_path", {}).get("processed_data", "processed_data")
        processed_train_filename = self.config.get("filenames", {}).get("processed_train", "processed_train.csv")
        processed_test_filename = self.config.get("filenames", {}).get("processed_test", "processed_test.csv")
        
        self.artifact_store.save_artifact(
            processed_train, subdir=processed_path, name=processed_train_filename
        )
        self.artifact_store.save_artifact(
            processed_test, subdir=processed_path, name=processed_test_filename
        )
        
        logging.info("Data preprocessing completed")
        
        self.outputs = {
            "processed_train": processed_train,
            "processed_test": processed_test
        }
        
        return processed_train, processed_test
    
    def _preprocess_data(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Apply preprocessing steps based on configuration."""
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Handle missing values
        if config.get("handle_missing", False):
            missing_strategy = config.get("missing_strategy", "drop")
            if missing_strategy == "drop":
                df = df.dropna()
            elif missing_strategy == "mean":
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif missing_strategy == "median":
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif missing_strategy == "mode":
                for col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0])
        
        # Handle categorical features
        if config.get("encode_categorical", False):
            categorical_cols = config.get("categorical_columns", [])
            if not categorical_cols:
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            encoding_strategy = config.get("encoding_strategy", "one_hot")
            if encoding_strategy == "one_hot":
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            elif encoding_strategy == "label":
                for col in categorical_cols:
                    df[col] = pd.factorize(df[col])[0]
        
        # Feature scaling
        if config.get("scale_features", False):
            scaling_strategy = config.get("scaling_strategy", "standard")
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if scaling_strategy == "standard":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            elif scaling_strategy == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # Feature selection
        if config.get("feature_selection", False):
            selected_features = config.get("selected_features", [])
            if selected_features:
                df = df[selected_features]
        
        return df


class ModelTrainingStep(Step):
    def __init__(self, config: Config, artifact_store: ArtifactStore):
        super().__init__(name="model_training")
        self.config = config
        self.artifact_store = artifact_store
        
    def execute(self, train_data: pd.DataFrame) -> Any:
        """Execute model training step."""
        self.inputs = {
            "train_data": train_data
        }
        
        # Get model configuration
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "linear_regression")
        
        # Prepare features and target
        target_column = model_config.get("target_column")
        if not target_column:
            raise ValueError("Target column must be specified in configuration")
        
        X = train_data.drop(columns=[target_column])
        y = train_data[target_column]
        
        # Train model based on type
        model = self._train_model(X, y, model_type, model_config)
        
        # Save model artifact
        models_path = self.config.get("folder_path", {}).get("models", "models")
        model_filename = self.config.get("filenames", {}).get("model", "model.pkl")
        
        self.artifact_store.save_artifact(
            model, subdir=models_path, name=model_filename
        )
        
        # Save feature names for inference
        feature_names = X.columns.tolist()
        features_filename = self.config.get("filenames", {}).get("features", "features.json")
        
        self.artifact_store.save_artifact(
            {"feature_names": feature_names},
            subdir=models_path,
            name=features_filename
        )
        
        logging.info(f"Model training completed. Model type: {model_type}")
        
        self.outputs = {
            "model": model,
            "feature_names": feature_names
        }
        
        return model
    
    def _train_model(self, X, y, model_type: str, config: Dict) -> Any:
        """Train a model based on the specified type and configuration."""
        if model_type == "linear_regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_type == "decision_tree":
            from sklearn.tree import DecisionTreeRegressor
            max_depth = config.get("params", {}).get("max_depth", None)
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            n_estimators = config.get("params", {}).get("n_estimators", 100)
            max_depth = config.get("params", {}).get("max_depth", None)
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        elif model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=42)
        elif model_type == "svm":
            from sklearn.svm import SVC
            kernel = config.get("params", {}).get("kernel", "rbf")
            model = SVC(kernel=kernel, random_state=42)
        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            n_estimators = config.get("params", {}).get("n_estimators", 100)
            model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X, y)
        return model


class ModelEvaluationStep(Step):
    def __init__(self, config: Config, artifact_store: ArtifactStore):
        super().__init__(name="model_evaluation")
        self.config = config
        self.artifact_store = artifact_store
        
    def execute(self, model: Any, test_data: pd.DataFrame) -> Dict[str, float]:
        """Execute model evaluation step."""
        self.inputs = {
            "model": model,
            "test_data": test_data
        }
        
        # Get model configuration
        model_config = self.config.get("model", {})
        target_column = model_config.get("target_column")
        task_type = model_config.get("task_type", "regression")
        
        # Prepare features and target
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics based on task type
        if task_type == "regression":
            metrics = self._evaluate_regression(y_test, y_pred)
        elif task_type == "classification":
            metrics = self._evaluate_classification(y_test, y_pred)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Save metrics artifact
        metrics_path = self.config.get("folder_path", {}).get("metrics", "metrics")
        metrics_filename = self.config.get("filenames", {}).get("metrics", "metrics.json")
        
        self.artifact_store.save_artifact(
            metrics, subdir=metrics_path, name=metrics_filename
        )
        
        logging.info(f"Model evaluation completed. Metrics: {metrics}")
        
        self.outputs = {
            "metrics": metrics
        }
        
        return metrics
    
    def _evaluate_regression(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate regression metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def _evaluate_classification(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate classification metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Check if binary or multiclass
        unique_labels = len(set(y_true.unique()))
        
        if unique_labels == 2:
            # Binary classification
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred)
            }
        else:
            # Multiclass classification
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_macro": precision_score(y_true, y_pred, average='macro'),
                "recall_macro": recall_score(y_true, y_pred, average='macro'),
                "f1_macro": f1_score(y_true, y_pred, average='macro')
            }
        
        return metrics


class Experiment:
    """Class to track experiments and their results."""
    
    def __init__(self, name: str, config: Config, artifact_store: ArtifactStore):
        self.name = name
        self.config = config
        self.artifact_store = artifact_store
        self.run_id = str(uuid.uuid4())
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{name}_{self.timestamp}"
        self.metrics = {}
        self.parameters = {}
        self.steps = {}
        
        # Create experiment directory
        experiment_path = self.config.get("folder_path", {}).get("experiments", "experiments")
        self.experiment_dir = os.path.join(experiment_path, self.experiment_id)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        logging.info(f"Created experiment '{name}' with ID {self.experiment_id}")
    
    def log_parameters(self, parameters: Dict[str, Any]):
        """Log parameters used in the experiment."""
        self.parameters.update(parameters)
        
        # Save parameters
        params_file = os.path.join(self.experiment_dir, "parameters.json")
        with open(params_file, "w") as f:
            json.dump(self.parameters, f, indent=2)
        
        logging.info(f"Logged parameters to {params_file}")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics from the experiment."""
        self.metrics.update(metrics)
        
        # Save metrics
        metrics_file = os.path.join(self.experiment_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        logging.info(f"Logged metrics to {metrics_file}")
    
    def register_step(self, step: Step):
        """Register a step with the experiment."""
        self.steps[step.name] = {
            "name": step.name,
            "inputs": step.inputs,
            "outputs": step.outputs
        }
        
        # Save updated steps info
        steps_file = os.path.join(self.experiment_dir, "steps.json")
        with open(steps_file, "w") as f:
            json.dump(self.steps, f, indent=2)
        
        logging.info(f"Registered step '{step.name}' with experiment '{self.name}'")


class Pipeline:
    """Main pipeline class that orchestrates the workflow."""

    def __init__(self, name: str, config_path: str):
        self.name = name
        self.config = Config.load_file(config_path)
        self.artifact_store = ArtifactStore(self.config)
        self.steps = []
        self.experiment = None
    
    def add_step(self, step: Step):
        """Add a step to the pipeline."""
        self.steps.append(step)
        logging.info(f"Added step '{step.name}' to pipeline '{self.name}'")
        return self
    
    def run(self, data_path: str, experiment_name: Optional[str] = None):
        """Execute the pipeline with all registered steps."""
        logging.info(f"Starting pipeline '{self.name}'")
        
        # Create an experiment if name provided
        if experiment_name:
            self.experiment = Experiment(experiment_name, self.config, self.artifact_store)
            # Log pipeline parameters
            self.experiment.log_parameters({
                "pipeline_name": self.name,
                "data_path": data_path,
                "config": self.config.config_dict
            })
        
        # Initialize step outputs
        step_outputs = {"data_path": data_path}
        
        # Execute each step in order
        for step in self.steps:
            logging.info(f"Executing step '{step.name}'")
            
            # Prepare arguments for the step
            step_args = self._prepare_step_args(step, step_outputs)
            
            # Execute the step
            outputs = step.execute(**step_args)
            
            # Handle different output types
            if isinstance(outputs, tuple):
                for i, output in enumerate(outputs):
                    step_outputs[f"{step.name}_output_{i}"] = output
            else:
                step_outputs[f"{step.name}_output"] = outputs
            
            # Copy outputs from step's own output dictionary
            for key, value in step.outputs.items():
                step_outputs[key] = value
            
            # Register step with experiment if available
            if self.experiment:
                self.experiment.register_step(step)
        
        # Log metrics if experiment is available and metrics were produced
        if self.experiment and "metrics" in step_outputs:
            self.experiment.log_metrics(step_outputs["metrics"])
        
        logging.info(f"Pipeline '{self.name}' completed successfully")
        return step_outputs
    
    def _prepare_step_args(self, step: Step, available_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare arguments for a step based on available outputs and step signature."""
        import inspect
        
        # Get the step's execute method signature
        signature = inspect.signature(step.execute)
        parameters = signature.parameters
        
        # Prepare arguments
        args = {}
        
        for name, param in parameters.items():
            if name in available_outputs:
                args[name] = available_outputs[name]
        
        return args


# Example usage
def create_ml_pipeline(config_path: str):
    # Load configuration
    config = Config.load_file(config_path)
    
    # Create pipeline
    pipeline = Pipeline("ml_training_pipeline", config_path)
    
    # Create artifact store
    artifact_store = ArtifactStore(config)
    
    # Add steps
    pipeline.add_step(DataIngestionStep(config, artifact_store))
    pipeline.add_step(DataPreprocessingStep(config, artifact_store))
    pipeline.add_step(ModelTrainingStep(config, artifact_store))
    pipeline.add_step(ModelEvaluationStep(config, artifact_store))
    
    return pipeline


if __name__ == "__main__":
    # Example configuration
    config_content = """
    base:
      test_size: 0.2
      random_state: 42
    
    folder_path:
      raw_data: raw_data
      processed_data: processed_data
      models: models
      metrics: metrics
      artifacts: artifacts
      experiments: experiments
    
    filenames:
      raw_train: train_data.csv
      raw_test: test_data.csv
      processed_train: processed_train.csv
      processed_test: processed_test.csv
      model: model.pkl
      features: features.json
      metrics: metrics.json
    
    preprocessing:
      handle_missing: true
      missing_strategy: mean
      encode_categorical: true
      encoding_strategy: one_hot
      scale_features: true
      scaling_strategy: standard
    
    model:
      type: random_forest
      task_type: regression
      target_column: target
      params:
        n_estimators: 100
        max_depth: 10
    """
    
    # Save example config
    os.makedirs("config", exist_ok=True)
    with open("config/config.yml", "w") as f:
        f.write(config_content)
    
    # Create sample data
    import numpy as np
    np.random.seed(42)
    X = np.random.rand(1000, 5)
    y = X[:, 0] * 2 + X[:, 1] * 3 - X[:, 2] * 1.5 + X[:, 3] * 0.5 - X[:, 4] + np.random.normal(0, 0.1, 1000)
    
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    df["target"] = y
    df.to_csv("data.csv", index=False)
    
    # Run pipeline
    data_path = "data.csv"
    config_path = "config/config.yml"
    
    pipeline = create_ml_pipeline(config_path)
    results = pipeline.run(data_path, experiment_name="sample_experiment")
    
    logging.info("Pipeline execution completed successfully!")
    if "metrics" in results:
        logging.info(f"Model metrics: {results['metrics']}")
