import os
import yaml
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn

# Prefect for orchestration
from prefect import task, flow
from prefect.task_runners import ConcurrentTaskRunner

# Flask for model deployment
from flask import Flask, request, jsonify

# Monitoring tools
import evidently
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, ModelPerformanceTab
from evidently.pipeline.column_mapping import ColumnMapping

# Prometheus client for metrics
from prometheus_client import start_http_server, Summary, Counter, Gauge

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Docker and cloud integration
import docker
from docker.errors import DockerException

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration loader class"""

    @staticmethod
    def load_file(config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise


class DataIngestion:
    """Data ingestion component"""

    @task
    def data_ingestion(
        self, path: str, config: Dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split data into training and test sets"""
        logger.info(f"Loading data from {path}")

        try:
            data = pd.read_csv(path)

            # Split data according to config
            train_size = config.get("data_split", {}).get("train_size", 0.8)
            target_column = config.get("data", {}).get("target_column")

            if not target_column:
                raise ValueError("Target column not specified in config")

            X = data.drop(columns=[target_column])
            y = data[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=42
            )

            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

            logger.info(
                f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}"
            )

            return train_data, test_data

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise


class FeatureEngineering:
    """Feature engineering component"""

    @task
    def process_features(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Process features according to configuration"""
        logger.info("Starting feature engineering")

        try:
            # Apply feature transformations based on config
            feature_config = config.get("features", {})
            processed_data = data.copy()

            # Handle categorical features
            categorical_features = feature_config.get("categorical_features", [])
            if categorical_features:
                processed_data = pd.get_dummies(
                    processed_data, columns=categorical_features
                )

            # Handle numerical features (example: scaling)
            # This could be expanded with more transformations

            logger.info(
                f"Feature engineering complete. Output shape: {processed_data.shape}"
            )
            return processed_data

        except Exception as e:
            logger.error(f"Error during feature engineering: {e}")
            raise


class ModelTraining:
    """Model training component"""

    @task
    def train_model(self, train_data: pd.DataFrame, config: Dict) -> Any:
        """Train machine learning model"""
        logger.info("Starting model training")

        try:
            # Extract model parameters from config
            model_config = config.get("model", {})
            model_type = model_config.get("type", "random_forest")
            target_column = config.get("data", {}).get("target_column")

            X = train_data.drop(columns=[target_column])
            y = train_data[target_column]

            # Initialize model based on type
            if model_type == "random_forest":
                n_estimators = model_config.get("n_estimators", 100)
                max_depth = model_config.get("max_depth", None)

                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=42
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Start MLflow run
            mlflow.start_run(run_name=config.get("experiment_name", "model_training"))

            # Log parameters
            for param, value in model_config.items():
                mlflow.log_param(param, value)

            # Train model
            model.fit(X, y)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # End MLflow run
            mlflow.end_run()

            logger.info("Model training complete")
            return model

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            mlflow.end_run()
            raise


class ModelEvaluation:
    """Model evaluation component"""

    @task
    def evaluate_model(self, model: Any, test_data: pd.DataFrame, config: Dict) -> Dict:
        """Evaluate trained model on test data"""
        logger.info("Starting model evaluation")

        try:
            target_column = config.get("data", {}).get("target_column")

            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1": f1_score(y_test, y_pred, average="weighted"),
            }

            # Log metrics with MLflow
            mlflow.start_run(run_name=config.get("experiment_name", "model_evaluation"))
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.end_run()

            logger.info(f"Model evaluation complete. Metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            mlflow.end_run()
            raise


import logging
from datetime import datetime
import pandas as pd
from src.core.oi import ArtifactStore, Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Deployment:
    """Handles model deployment operations"""

    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)

    def deploy_model(self, model):
        """Deploy the best model to production"""
        deploy_path = self.config.get("model", {}).get("deployment_path", "deployment")
        self.artifact_store.save_artifact(
            model, "production_model", subdir=deploy_path, format="pkl"
        )
        logging.info(f"Model deployed to {deploy_path}/production_model.pkl")


class Monitoring:
    """Handles model performance monitoring"""

    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)

    def log_performance(self, metrics):
        """Log model performance metrics"""
        monitor_path = self.config.get("model_monitor", {}).get(
            "monitor_path", "monitoring"
        )
        self.artifact_store.save_artifact(
            pd.DataFrame([metrics]),
            f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            subdir=monitor_path,
            format="csv",
        )
        logging.info("Performance metrics logged")


class Retraining:
    """Handles model retraining logic"""

    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)

    def check_retraining(self, current_accuracy):
        """Check if retraining is needed based on performance drift"""
        try:
            # Load previous performance metrics
            monitor_path = self.config.get("model_monitor", {}).get(
                "monitor_path", "monitoring"
            )
            metrics = self.artifact_store.load_artifact(
                "performance_latest", subdir=monitor_path, format="csv"
            )

            if metrics is not None:
                prev_accuracy = metrics["accuracy"].iloc[-1]
                drift = abs(current_accuracy - prev_accuracy)

                if drift > self.config.get("model_monitor", {}).get(
                    "drift_threshold", 0.05
                ):
                    logging.info(
                        f"Significant accuracy drift detected: {drift:.2f}. Retraining needed."
                    )
                    return True

            logging.info("No significant performance drift detected")
            return False

        except Exception as e:
            logging.error(f"Error checking retraining need: {e}")
            return False


class Stack:
    """Integrated MLOps component stack"""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

        # Initialize components
        self.deployment = Deployment(config)
        self.monitoring = Monitoring(config)
        self.retraining = Retraining(config)

        logging.info(
            f"Initialized MLOps stack '{name}' with deployment, monitoring, and retraining capabilities"
        )


class TrainingPipeline:
    """End-to-end ML pipeline with deployment and monitoring"""

    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_file("config/config.yml").config_dict
        self.stack = Stack("prod_ml_stack", self.config)
        self.artifact_store = ArtifactStore(self.config)

    def run(self):
        try:
            # [Existing pipeline steps...]
            # After model training and evaluation:

            # 1. Deployment
            best_model = joblib.load("artifacts/model/best_model.pkl")
            self.stack.deployment.deploy_model(best_model)

            # 2. Monitoring
            test_metrics = {
                "accuracy": 0.999,  # Replace with actual metrics
                "precision": 0.995,
                "recall": 0.998,
                "timestamp": datetime.now().isoformat(),
            }
            self.stack.monitoring.log_performance(test_metrics)

            # 3. Retraining check
            if self.stack.retraining.check_retraining(test_metrics["accuracy"]):
                logging.info("Initiating retraining pipeline...")
                # Add retraining logic here

            return "Pipeline execution completed successfully"

        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            raise


class ModelDeployment:
    """Model deployment component using Flask"""

    def __init__(self):
        self.app = Flask(__name__)
        self.model = None
        self.config = None

    def setup_routes(self):
        """Set up Flask routes for prediction"""

        @self.app.route("/health", methods=["GET"])
        def health_check():
            return jsonify({"status": "healthy"})

        @self.app.route("/predict", methods=["POST"])
        def predict():
            try:
                # Get data from request
                data = request.json

                # Convert to DataFrame
                input_df = pd.DataFrame([data])

                # Preprocess input (simplified)
                # In a real system, this would apply the same preprocessing as training

                # Make prediction
                prediction = self.model.predict(input_df)[0]

                # Record prediction for monitoring
                self.prediction_counter.inc()
                self.prediction_latency.observe(1.0)  # Placeholder

                return jsonify(
                    {
                        "prediction": (
                            prediction.tolist()
                            if isinstance(prediction, np.ndarray)
                            else prediction
                        ),
                        "model_version": self.config.get("model", {}).get(
                            "version", "unknown"
                        ),
                    }
                )

            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                return jsonify({"error": str(e)}), 500

    @task
    def deploy_model(self, model: Any, config: Dict) -> bool:
        """Deploy trained model with Flask"""
        logger.info("Starting model deployment")

        try:
            self.model = model
            self.config = config

            # Set up Prometheus metrics
            self.prediction_counter = Counter(
                "predictions_total", "Total number of predictions"
            )
            self.prediction_latency = Summary(
                "prediction_latency_seconds", "Prediction latency in seconds"
            )

            # Start Prometheus metrics server
            metrics_port = config.get("monitoring", {}).get("prometheus_port", 9090)
            start_http_server(metrics_port)

            # Setup Flask routes
            self.setup_routes()

            # Start Flask server in a separate thread
            flask_port = config.get("deployment", {}).get("flask_port", 5000)
            from threading import Thread

            server_thread = Thread(
                target=lambda: self.app.run(
                    host="0.0.0.0", port=flask_port, debug=False, use_reloader=False
                )
            )
            server_thread.daemon = True
            server_thread.start()

            logger.info(f"Model deployed successfully at http://localhost:{flask_port}")
            return True

        except Exception as e:
            logger.error(f"Error during model deployment: {e}")
            return False


class ModelMonitoring:
    """Model monitoring component using Evidently and Grafana"""

    def __init__(self):
        self.reference_data = None
        self.current_data = []
        self.column_mapping = None
        self.dashboard = None

    @task
    def setup_monitoring(self, reference_data: pd.DataFrame, config: Dict) -> str:
        """Set up monitoring dashboard"""
        logger.info("Setting up model monitoring")

        try:
            self.reference_data = reference_data

            target_column = config.get("data", {}).get("target_column")
            numerical_features = config.get("features", {}).get(
                "numerical_features", []
            )
            categorical_features = config.get("features", {}).get(
                "categorical_features", []
            )

            # Set up column mapping for Evidently
            self.column_mapping = ColumnMapping(
                target=target_column,
                prediction="prediction",
                numerical_features=numerical_features,
                categorical_features=categorical_features,
            )

            # Create Evidently dashboard
            self.dashboard = Dashboard(tabs=[DataDriftTab(), ModelPerformanceTab()])

            # Create dashboard output directory
            dashboard_path = os.path.join(
                config.get("monitoring", {}).get("dashboard_path", "monitoring"),
                "dashboard.html",
            )
            os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)

            # Initialize with reference data
            self.update_dashboard(dashboard_path)

            logger.info(f"Monitoring dashboard created at {dashboard_path}")
            return dashboard_path

        except Exception as e:
            logger.error(f"Error during monitoring setup: {e}")
            raise

    def collect_prediction_data(self, input_data: Dict, prediction: Any):
        """Collect prediction data for monitoring"""
        data_record = input_data.copy()
        data_record["prediction"] = prediction
        self.current_data.append(data_record)

    def update_dashboard(self, dashboard_path: str) -> None:
        """Update monitoring dashboard with new data"""
        if len(self.current_data) > 0:
            current_df = pd.DataFrame(self.current_data)
            self.dashboard.calculate(
                self.reference_data, current_df, column_mapping=self.column_mapping
            )
            self.dashboard.save(dashboard_path)


class ModelRetraining:
    """Model retraining component"""

    @task
    def check_retraining_trigger(self, metrics: Dict, config: Dict) -> bool:
        """Check if retraining is needed based on metrics"""
        logger.info("Checking retraining trigger")

        try:
            retraining_config = config.get("retraining", {})
            metric_threshold = retraining_config.get("metric_threshold", {})

            # Check if any metric falls below threshold
            for metric_name, threshold in metric_threshold.items():
                if metric_name in metrics and metrics[metric_name] < threshold:
                    logger.info(
                        f"Retraining triggered: {metric_name}={metrics[metric_name]} below threshold {threshold}"
                    )
                    return True

            # Check time-based retraining
            # (This would require tracking when the model was last trained)

            logger.info("No retraining needed at this time")
            return False

        except Exception as e:
            logger.error(f"Error during retraining check: {e}")
            return False

    @task
    def retrain_model(self, path: str, config: Dict) -> bool:
        """Retrain model with fresh data"""
        logger.info("Starting model retraining")

        try:
            # Load fresh data
            # For simplicity, we're using the same data path
            # In a real system, this might fetch new production data

            # Create a new training pipeline
            training_pipeline = TrainingPipeline(path)

            # Run training
            training_pipeline.run()

            # Optionally deploy the new model
            # (This would involve stopping the current deployment and starting a new one)

            logger.info("Model retraining complete")
            return True

        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            return False


class Stack:
    """Integrated MLOps component stack"""

    def __init__(
        self,
        name: str,
        max_workers: int = 4,
        running: str = "local",
        scaler: str = "local",
    ):
        self.name = name
        self.max_workers = max_workers
        self.running = running
        self.scaler = scaler
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Initialize components
        self.data_ingestion = DataIngestion()
        self.feature_engineering = FeatureEngineering()
        self.model_training = ModelTraining()
        self.model_evaluation = ModelEvaluation()
        self.deployment = ModelDeployment()
        self.monitoring = ModelMonitoring()
        self.retraining = ModelRetraining()

    def train(self, train_data: pd.DataFrame, config: Dict) -> Any:
        """Execute training pipeline"""
        processed_data = self.feature_engineering.process_features(train_data, config)
        model = self.model_training.train_model(processed_data, config)
        return model

    def evaluate(self, model: Any, test_data: pd.DataFrame, config: Dict) -> Dict:
        """Execute evaluation pipeline"""
        processed_test_data = self.feature_engineering.process_features(
            test_data, config
        )
        metrics = self.model_evaluation.evaluate_model(
            model, processed_test_data, config
        )
        return metrics

    def deploy(self, model: Any, config: Dict) -> bool:
        """Execute deployment pipeline"""
        deployment_success = self.deployment.deploy_model(model, config)
        return deployment_success

    def monitor(self, reference_data: pd.DataFrame, config: Dict) -> str:
        """Execute monitoring pipeline"""
        dashboard_path = self.monitoring.setup_monitoring(reference_data, config)
        return dashboard_path

    def check_and_retrain(self, metrics: Dict, path: str, config: Dict) -> bool:
        """Execute retraining pipeline if needed"""
        retraining_needed = self.retraining.check_retraining_trigger(metrics, config)

        if retraining_needed:
            retraining_success = self.retraining.retrain_model(path, config)
            return retraining_success

        return False


class TrainingPipeline:
    """Main ML pipeline orchestrator"""

    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_file("config/config.yml")

    @flow(name="MLOps Pipeline", task_runner=ConcurrentTaskRunner())
    def run(self) -> Dict:
        """Run the complete MLOps pipeline"""
        logger.info(f"Starting MLOps pipeline for data at {self.path}")

        try:
            # Initialize stack
            option_run = self.config.get("execution", {}).get("option_run", "local")
            option_scale = self.config.get("execution", {}).get("option_scale", "local")

            my_stack = Stack(
                "ml_pipeline", max_workers=4, running=option_run, scaler=option_scale
            )

            # Data ingestion
            train_data, test_data = my_stack.data_ingestion.data_ingestion(
                self.path, self.config
            )

            # Model training
            model = my_stack.train(train_data, self.config)

            # Model evaluation
            metrics = my_stack.evaluate(model, test_data, self.config)

            # Model deployment
            deployment_success = my_stack.deploy(model, self.config)

            # Set up monitoring
            dashboard_path = my_stack.monitor(test_data, self.config)

            # Check if retraining is needed
            retraining_success = my_stack.check_and_retrain(
                metrics, self.path, self.config
            )

            # Return results
            results = {
                "deployment": deployment_success,
                "monitoring": dashboard_path,
                "re-training": retraining_success,
            }

            logger.info(f"MLOps pipeline completed successfully. Results: {results}")
            return results

        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise


def main():
    """Main entry point"""
    # Create and run pipeline
    path = "data/data.csv"
    pipeline = TrainingPipeline(path)
    results = pipeline.run()
    print(f"Pipeline Results: {results}")


if __name__ == "__main__":
    main()















#config.yml
folder_path: 
  artifacts: "artifacts"
  raw_path: "raw"
  train: "train.csv"
  test: "test.csv"

import pandas as pd
import os
import yaml
import json
import pickle
import logging
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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
        self.base_path = self.config.get("folder_path", {}).get("artifacts", "artifacts")
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Artifact store initialized at '{self.base_path}'")

    def save_artifact(self, artifact: Any, name: str, subdir: str) -> None:
        """Save an artifact in the specified format."""
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

    def load_artifact(self, name: str, subdir: str):
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


class Stack:
    """Integrated MLOps component stack"""
    
    def __init__(self, name: str, config: dict = None, max_workers: int = 4):
        self.name = name
        self.config = config
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # Add components as attributes
        self.dataingest = DataIngestion()
        logging.info(f"Initialized Stack '{name}' with {max_workers} workers")
        
    def run_parallel(self, func, *args):
        """Run a function in parallel using the thread pool"""
        future = self.executor.submit(func, *args)
        return future
    
    def shutdown(self, wait=True):
        """Shutdown the executor"""
        self.executor.shutdown(wait=wait)
        logging.info(f"Stack '{self.name}' executor shutdown")


class DataIngestion:
    """Handles data ingestion process"""

    def __init__(self,config):
        self.config = config
        self.artifact_store = ArtifactStore(config)

    def data_ingestion(self, path):
        """split the input data"""
        # Check if artifacts already exist

        raw_path = self.config.get("folder_path", {}).get("raw_path", {})
        train_filename = self.config.get("folder_path", {}).get("train", {})
        test_filename = self.config.get("folder_path", {}).get("test", {})

        
        self.artifact_store.load_artifact(
                train_data,
                name=train_filename,
                subdir=raw_path
            )
            
        self.artifact_store.load_artifact(
            test_data,
            name=test_filename,
            subdir=raw_path
            )

        if train_data is not None and test_data is not None:
            logging.info("Loaded raw artifacts from store. Skipping data ingestion.")
            return train_data, test_data
        try:

            # Load raw data
            df = pd.read_csv(path)

            # Split data
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
            
            # Get configuration
            raw_path = self.config.get("folder_path", {}).get("raw_path", {})
            train_filename = self.config.get("folder_path", {}).get("train", {})
            test_filename = self.config.get("folder_path", {}).get("test", {})

            # Save raw artifacts
            self.artifact_store.save_artifact(
                train_data,
                name=train_filename,
                subdir=raw_path
            )
            
            self.artifact_store.save_artifact(
                test_data,
                name=test_filename,
                subdir=raw_path
            )

            logging.info("Data ingestion completed successfully")
            return train_data, test_data

        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise


class TrainingPipeline:
    """Main ML pipeline orchestrator."""

    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_file("config/config.yml").config_dict
        self.stack = Stack("ml_pipeline", self.config, max_workers=4)

    def run(self):
        logging.info(f"Starting pipeline for data at {self.path}")

        try:
            # Run data ingestion
            train_data, test_data = self.stack.dataingest.data_ingestion(self.path, self.config)
            
            # Add more pipeline steps here as needed
            
            logging.info("Pipeline completed successfully")
            return "pipeline_success"
        
        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            return None

if __name__ == "__main__":
    path = "data/churn-train.csv"
    pipeline = TrainingPipeline(path)
    result = pipeline.run()
    
    if result is not None:
        print("Pipeline executed successfully!")
    else:
        print("Pipeline execution failed")




from src.ml_customer_churn.data_ingestion import DataIngestion
from src.ml_customer_churn.data_validation import DataValidation
from src.ml_customer_churn.data_transformation import DataTransformation
from src.ml_customer_churn.model_trainer import ModelTrainer
from src.ml_customer_churn.model_evaluation import ModelEvaluation

import yaml
import json
import os
import logging
from typing import Dict, Any, List, Optional
import pickle
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ArtifactStore:
    """Stores and retrieves intermediate artifacts for the pipeline."""

    def __init__(self, config: Dict):
        self.config = config
        self.base_path = self.config.get("folder", {}).get("output_path", "artifacts")
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Artifact store initialized at '{self.base_path}'")

    def save_artifact(self, artifact: Any, name: str, subdir: str) -> None:
        """Save an artifact in the specified format."""
        artifact_dir = os.path.join(self.base_path, subdir)
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, f"{name}")

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

    def load_artifact(self, name: str, subdir: str):
        """Load an artifact in the specified format."""
        artifact_path = os.path.join(self.base_path, subdir, f"{name}")
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

    def list_artifacts(self) -> list:
        """
        List all artifacts in the store.
        Returns:
            list: List of artifact names.
        """
        artifacts = []

        for subdir, _, files in os.walk(self.base_path):
            for file in files:
                artifacts.append(
                    os.path.join(os.path.relpath(subdir, self.base_path), file)
                )

        logging.info(f"Artifacts in store: {artifacts}")
        return artifacts


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


class Stack:
    """Integrated MLOps component stack"""

    def __init__(self, name: str, config: dict = None, max_workers: int = 4):
        self.name = name
        self.config = config
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logging.info(f"Initialized Stack '{name}' with {max_workers} workers")

        # Initialize components
        self.dataingest = DataIngestion(config) if config else None
        self.datavalidate = DataValidation(config) if config else None
        # Uncomment these as you implement them
        # self.datatransformer = DataTransformation(config) if config else None
        # self.modeltrainer = ModelTrainer(config) if config else None
        # self.modelevaluation = ModelEvaluation(config) if config else None
        # self.deployment = ModelDeployment(config) if config else None
        # self.monitoring = ModelMonitoring(config) if config else None
        # self.retraining = ModelRetraining(config) if config else None

    def run_parallel(self, func, *args):
        """Run a function in parallel using the thread pool"""
        future = self.executor.submit(func, *args)
        return future

    def shutdown(self, wait=True):
        """Shutdown the executor"""
        self.executor.shutdown(wait=wait)
        logging.info(f"Stack '{self.name}' executor shutdown")


class DataIngestion:
    """Handles data ingestion and artifact storage."""

    def __init__(self, config: Dict):
        self.config = config
        self.artifact_store = ArtifactStore(self.config)

    def data_ingestion(self, path: str) -> tuple:
        """
        Load and split data into training and test sets, and save artifacts.
        If artifacts already exist, load them instead of reprocessing.

        Args:
            path (str): Path to the raw data file.

        Returns:
            tuple: (train_data, test_data) as pandas DataFrames.
        """
        # Check if artifacts already exist
        train_data = self.artifact_store.load_artifact(
            "train.csv", subdir=self.config.get("folder", {}).get("raw_path", "raw")
        )
        test_data = self.artifact_store.load_artifact(
            "test.csv", subdir=self.config.get("folder", {}).get("raw_path", "raw")
        )

        if train_data is not None and test_data is not None:
            logger.info("Loaded raw artifacts from store. Skipping data ingestion.")
            return train_data, test_data

        try:
            # Load raw data
            df = pd.read_csv(path)

            # Split data
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            # Save raw artifacts
            self.artifact_store.save_artifact(
                train_data,
                "train.csv",
                subdir=self.config.get("folder", {}).get("raw_path", "raw"),
            )
            self.artifact_store.save_artifact(
                test_data,
                "test.csv",
                subdir=self.config.get("folder", {}).get("raw_path", "raw"),
            )

            logger.info("Data ingestion completed")
            return train_data, test_data

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise


class TrainingPipeline:
    """Main ML pipeline orchestrator."""

    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_file("config/config.yml").config_dict
        self.artifact_store = ArtifactStore(self.config)
        self.stack = Stack("prod_ml_stack", self.config, max_workers=4)

    def run(self):
        logging.info(f"Starting pipeline for data at {self.path}")

        try:
            # Data ingestion - run synchronously first to ensure data is available
            logging.info("Task 1: Data ingestion")
            train_data, test_data = self.stack.dataingest.data_ingestion(self.path)

            # Data validation - can be run in parallel with subsequent steps if needed
            logging.info("Task 2: Data validation")
            val_future = self.stack.run_parallel(
                self.stack.datavalidate.data_validation, train_data, test_data
            )

            # Wait for validation to complete
            val_train_data, val_test_data = val_future.result()

            # Uncomment as you implement these components
            # # Data transformation
            # logging.info("Task 3: Data transformation")
            # transform_future = self.stack.run_parallel(
            #     self.stack.datatransformer.data_transformation,
            #     val_train_data,
            #     val_test_data
            # )
            # xtrain, xtest, ytrain, ytest = transform_future.result()

            # # Model training
            # logging.info("Task 4: Model training")
            # train_future = self.stack.run_parallel(
            #     self.stack.modeltrainer.model_trainer,
            #     xtrain,
            #     xtest,
            #     ytrain,
            #     ytest
            # )
            # model = train_future.result()

            # # Model evaluation
            # logging.info("Task 5: Model evaluation")
            # eval_future = self.stack.run_parallel(
            #     self.stack.modelevaluation.model_evaluation,
            #     model,
            #     xtest,
            #     ytest
            # )
            # eval_results = eval_future.result()

            # Clean up resources
            self.stack.shutdown()

            logging.info("Pipeline execution completed.")
            return "Pipeline execution completed."

        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            self.stack.shutdown(wait=False)  # Force shutdown in case of error
            return None


if __name__ == "__main__":
    path = "data/churn-train.csv"
    pipeline = TrainingPipeline(path)
    # results = pipeline.run()
    # print(f"Pipeline Results: {results}")
    if pipeline.run() is not None:
        print("Pipeline executed successfully!")
        # Uncomment to check retraining status
        # retraining_status = pipeline.pipeline_stack.get_artifact("retraining_decision")
        # if retraining_status and retraining_status.data.get('required'):
        #     print("System requires retraining based on:")
        #     print(json.dumps(retraining_status.data, indent=2))
    else:
        print("Pipeline execution failed")



from src.ml_customer_churn.data_ingestion import DataIngestion
from src.ml_customer_churn.data_validation import DataValidation
from src.ml_customer_churn.data_transformation import DataTransformation
from src.ml_customer_churn.model_trainer import ModelTrainer
from src.ml_customer_churn.model_evaluation import ModelEvaluation

import yaml
import json
import os
import logging
from typing import Dict, Any, List, Optional
import pickle
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ArtifactStore:
    """Stores and retrieves intermediate artifacts for the pipeline."""

    def __init__(self, config: Dict):
        self.config = config
        self.base_path = self.config.get("folder", {}).get("output_path", "artifacts")
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Artifact store initialized at '{self.base_path}'")

    def save_artifact(
        self, artifact: Any, name: str, subdir: str) -> None:
        """Save an artifact in the specified format."""
        artifact_dir = os.path.join(self.base_path, subdir)
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, f"{name}")

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

    def load_artifact(
        self, name: str, subdir: str):
        """Load an artifact in the specified format."""
        artifact_path = os.path.join(self.base_path, subdir, f"{name}")
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

    def list_artifacts(self) -> list:
        """
        List all artifacts in the store.
        Returns:
            list: List of artifact names.
        """
        artifacts = []
        
        for subdir, _, files in os.walk(self.base_path):
            for file in files:
                artifacts.append(os.path.join(os.path.relpath(subdir, self.base_path), file))
        
        logging.info(f"Artifacts in store: {artifacts}")
        return artifacts


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


class Stack:
    """Integrated MLOps component stack"""
    
    def __init__(self, name: str, config: dict = None, max_workers: int = 4):
        self.name = name
        self.config = config
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logging.info(f"Initialized Stack '{name}' with {max_workers} workers")
        
        # Initialize components
        self.dataingest = DataIngestion(config) if config else None
        self.datavalidate = DataValidation(config) if config else None
        # Uncomment these as you implement them
        # self.datatransformer = DataTransformation(config) if config else None
        # self.modeltrainer = ModelTrainer(config) if config else None
        # self.modelevaluation = ModelEvaluation(config) if config else None
        # self.deployment = ModelDeployment(config) if config else None
        # self.monitoring = ModelMonitoring(config) if config else None
        # self.retraining = ModelRetraining(config) if config else None
    
    def run_parallel(self, func, *args):
        """Run a function in parallel using the thread pool"""
        future = self.executor.submit(func, *args)
        return future
    
    def shutdown(self, wait=True):
        """Shutdown the executor"""
        self.executor.shutdown(wait=wait)
        logging.info(f"Stack '{self.name}' executor shutdown")


class DataIngestion:
    """Handles data ingestion and artifact storage."""

    def __init__(self, config: Dict):
        self.config = config
        self.artifact_store = ArtifactStore(self.config)

    def data_ingestion(self, path: str) -> tuple:
        """
        Load and split data into training and test sets, and save artifacts.
        If artifacts already exist, load them instead of reprocessing.

        Args:
            path (str): Path to the raw data file.

        Returns:
            tuple: (train_data, test_data) as pandas DataFrames.
        """
        # Check if artifacts already exist
        train_data = self.artifact_store.load_artifact(
            "train.csv",
            subdir=self.config.get("folder", {}).get("raw_path", "raw")
        )
        test_data = self.artifact_store.load_artifact(
            "test.csv",
            subdir=self.config.get("folder", {}).get("raw_path", "raw")
        )

        if train_data is not None and test_data is not None:
            logger.info("Loaded raw artifacts from store. Skipping data ingestion.")
            return train_data, test_data

        try:
            # Load raw data
            df = pd.read_csv(path)

            # Split data
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            # Save raw artifacts
            self.artifact_store.save_artifact(
                train_data,
                "train.csv",
                subdir=self.config.get("folder", {}).get("raw_path", "raw")
            )
            self.artifact_store.save_artifact(
                test_data,
                "test.csv",
                subdir=self.config.get("folder", {}).get("raw_path", "raw")
            )

            logger.info("Data ingestion completed")
            return train_data, test_data

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise


class TrainingPipeline:
    """Main ML pipeline orchestrator."""

    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_file("config/config.yml").config_dict
        self.artifact_store = ArtifactStore(self.config)
        self.stack = Stack("prod_ml_stack", self.config, max_workers=4)

    def run(self):
        logging.info(f"Starting pipeline for data at {self.path}")

        try:
            # Data ingestion - run synchronously first to ensure data is available
            logging.info("Task 1: Data ingestion")
            train_data, test_data = self.stack.dataingest.data_ingestion(self.path)

            # Data validation - can be run in parallel with subsequent steps if needed
            logging.info("Task 2: Data validation")
            val_future = self.stack.run_parallel(
                self.stack.datavalidate.data_validation,
                train_data, 
                test_data
            )
            
            # Wait for validation to complete
            val_train_data, val_test_data = val_future.result()

            # Uncomment as you implement these components
            # # Data transformation
            # logging.info("Task 3: Data transformation")
            # transform_future = self.stack.run_parallel(
            #     self.stack.datatransformer.data_transformation,
            #     val_train_data, 
            #     val_test_data
            # )
            # xtrain, xtest, ytrain, ytest = transform_future.result()

            # # Model training
            # logging.info("Task 4: Model training")
            # train_future = self.stack.run_parallel(
            #     self.stack.modeltrainer.model_trainer,
            #     xtrain, 
            #     xtest, 
            #     ytrain, 
            #     ytest
            # )
            # model = train_future.result()

            # # Model evaluation
            # logging.info("Task 5: Model evaluation")
            # eval_future = self.stack.run_parallel(
            #     self.stack.modelevaluation.model_evaluation,
            #     model, 
            #     xtest, 
            #     ytest
            # )
            # eval_results = eval_future.result()

            # Clean up resources
            self.stack.shutdown()
            
            logging.info("Pipeline execution completed.")
            return "Pipeline execution completed."
            
        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            self.stack.shutdown(wait=False)  # Force shutdown in case of error
            return None


if __name__ == "__main__":
    path = "data/churn-train.csv"
    pipeline = TrainingPipeline(path)
    # results = pipeline.run()
    # print(f"Pipeline Results: {results}")
    if pipeline.run() is not None:
        print("Pipeline executed successfully!")
        # Uncomment to check retraining status
        # retraining_status = pipeline.pipeline_stack.get_artifact("retraining_decision")
        # if retraining_status and retraining_status.data.get('required'):
        #     print("System requires retraining based on:")
        #     print(json.dumps(retraining_status.data, indent=2))
    else:
        print("Pipeline execution failed")



# class DataValidation:

#     def __init__(self, config):
#         self.config = config
#         self.artifact_store = ArtifactStore(self.config)

#     # @task
#     def data_validation(self, train_data, test_data):#, config):
#         try:

#             val_train_data = self.artifact_store.load_artifact(
#             "val_train_data",
#             subdir=self.config.get("data_validate", {}).get("validate_path", "validate"),
#             format="csv",
#             )
#             val_test_data = self.artifact_store.load_artifact(
#                 "val_test_data",
#                 subdir=self.config.get("data_validate", {}).get("validate_path", "validate"),
#                 format="csv",
#             )

#             if val_train_data is not None and val_test_data is not None:
#                 logger.info("Loaded validation artifacts from store. Skipping data validation.")
#                 return val_train_data, val_test_data


#             for feature in train_data.columns:
#                 ks_stat, p_value = ks_2samp(train_data[feature], test_data[feature])

#                 if p_value < 0.05:
#                     print(
#                         f"Data drift detected for feature {feature} with p-value: {p_value}"
#                     )
#                     return "Error drift data"

#             val_train_data = train_data
#             val_test_data = test_data


#             self.artifact_store.save_artifact(
#                 val_train_data,
#                 "val_train_data",
#                 subdir=self.config.get("data_validate", {}).get("validate_path", "validate"),
#                 format="csv",
#             )
#             self.artifact_store.save_artifact(
#                 val_test_data,
#                 "val_test_data",
#                 subdir=self.config.get("data_validate", {}).get("validate_path", "validate"),
#                 format="csv",
#             )
#             # validate_path = config.get("data_path", {}).get(
#             #     "data"
#             # )  # Using the passed config instance
#             # vali_folder = config.get("data_validate", {}).get("validate_path")
#             # train_filename = config.get("data_validate", {}).get("train_val")
#             # test_filename = config.get("data_validate", {}).get("test_val")

#             # validate_data_path = os.path.join(validate_path, vali_folder)
#             # os.makedirs(validate_data_path, exist_ok=True)
#             # # save train test to csv
#             # val_train = train_data
#             # val_test = test_data

#             # val_train.to_csv(
#             #     os.path.join(validate_data_path, str(train_filename)),
#             #     index=False,
#             # )
#             # val_test.to_csv(
#             #     os.path.join(validate_data_path, str(test_filename)),
#             #     index=False,
#             # )

#             logging.info(f"Data validation completed")
#             return val_train_data, val_test_data

#         except Exception as e:
#             raise e




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import os
import joblib
import logging
from src.core.oi import ArtifactStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class DataTransformation:
    """Handles data transformation and artifact storage."""

    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(self.config)

    @staticmethod
    def remove_outliers(data, labels):
        """Replaces outliers in numerical columns with the median."""
        for label in labels:
            q1 = data[label].quantile(0.25)
            q3 = data[label].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            lower_bound = q1 - 1.5 * iqr
            data[label] = data[label].mask(
                data[label] < lower_bound, data[label].median(), axis=0
            )
            data[label] = data[label].mask(
                data[label] > upper_bound, data[label].median(), axis=0
            )
        return data

    def data_transformation(self, val_train, val_test):

        # Check if transformation artifacts already exist
        X_train = self.artifact_store.load_artifact(
            "xtrain",
            subdir=self.config.get("data_transformer", {}).get(
                "transform_path", "transformer"
            ),
            format="csv",
        )
        X_test = self.artifact_store.load_artifact(
            "xtest",
            subdir=self.config.get("data_transformer", {}).get(
                "transform_path", "transformer"
            ),
            format="csv",
        )
        y_train = self.artifact_store.load_artifact(
            "ytrain",
            subdir=self.config.get("data_transformer", {}).get(
                "transform_path", "transformer"
            ),
            format="csv",
        )
        y_test = self.artifact_store.load_artifact(
            "ytest",
            subdir=self.config.get("data_transformer", {}).get(
                "transform_path", "transformer"
            ),
            format="csv",
        )

        if (
            X_train is not None
            and X_test is not None
            and y_train is not None
            and y_test is not None
        ):
            logging.info(
                "Loaded transformation artifacts from store. Skipping data transformation."
            )
            return X_train, X_test, y_train, y_test

        try:
            # Remove outliers from numerical columns
            val_train = self.remove_outliers(
                val_train, val_train.select_dtypes(include=["int64", "float64"]).columns
            )
            val_test = self.remove_outliers(
                val_test, val_test.select_dtypes(include=["int64", "float64"]).columns
            )

            # Prepare features and target
            xtrain_data = val_train.drop(columns=["Churn", "CustomerID"], axis=1)
            xtest_data = val_test.drop(columns=["Churn", "CustomerID"], axis=1)
            y_train = val_train["Churn"]
            y_train.fillna(y_train.mode()[0], inplace=True)
            y_test = val_test["Churn"]
            y_test.fillna(y_train.mode()[0], inplace=True)

            # Identify numerical and categorical features
            num_features = xtrain_data.select_dtypes(
                include=["int64", "float64"]
            ).columns
            cat_features = xtrain_data.select_dtypes(include=["object"]).columns

            # Define transformation pipelines
            num_transformer = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_transformer = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            # Create a ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num", num_transformer, num_features),
                    ("cat", cat_transformer, cat_features),
                ],
                remainder="passthrough",
            )

            # Fit on training data and transform both train & test sets
            xtrain_preprocessed = preprocessor.fit_transform(xtrain_data)
            xtest_preprocessed = preprocessor.transform(xtest_data)

            # Get transformed column names
            cat_columns = (
                preprocessor.transformers_[1][1]
                .named_steps["encoder"]
                .get_feature_names_out(cat_features)
            )
            new_columns = list(num_features) + list(cat_columns)

            # Convert transformed data into DataFrame
            X_train = pd.DataFrame(xtrain_preprocessed, columns=new_columns)
            X_test = pd.DataFrame(xtest_preprocessed, columns=new_columns)

            # Reset index for y_train and y_test to prevent misalignment
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            # Convert y_train and y_test to DataFrames
            y_train_df = pd.DataFrame(y_train, columns=["Churn"])
            y_test_df = pd.DataFrame(y_test, columns=["Churn"])

            # Save transformation artifacts
            self.artifact_store.save_artifact(
                X_train,
                "xtrain",
                subdir=self.config.get("data_transformer", {}).get(
                    "transform_path", "transformer"
                ),
                format="csv",
            )
            self.artifact_store.save_artifact(
                X_test,
                "xtest",
                subdir=self.config.get("data_transformer", {}).get(
                    "transform_path", "transformer"
                ),
                format="csv",
            )
            self.artifact_store.save_artifact(
                y_train_df,
                "ytrain",
                subdir=self.config.get("data_transformer", {}).get(
                    "transform_path", "transformer"
                ),
                format="csv",
            )
            self.artifact_store.save_artifact(
                y_test_df,
                "ytest",
                subdir=self.config.get("data_transformer", {}).get(
                    "transform_path", "transformer"
                ),
                format="csv",
            )

            # Save the transformer model for later use
            transformer_pkl = self.config.get("data_transformer", {}).get("transformer")
            transformer_data_path = os.path.join(
                self.config.get("data_path", {}).get("data", "artifacts"),
                self.config.get("data_transformer", {}).get(
                    "transform_path", "transformer"
                ),
            )
            os.makedirs(transformer_data_path, exist_ok=True)
            joblib.dump(
                preprocessor, os.path.join(transformer_data_path, transformer_pkl)
            )

            logging.info("Data transformation completed successfully.")
            return X_train, X_test, y_train_df, y_test_df

        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise
