# Initialize the framework
framework = MLOpsFramework("config/config.yml")

# Create a pipeline
pipeline = framework.create_pipeline("training_pipeline")

# Add steps
pipeline.add_step(framework.data_ingestion_step())
pipeline.add_step(framework.feature_engineering_step(), dependencies=["data_ingestion"])
pipeline.add_step(framework.model_training_step(), dependencies=["feature_engineering"])
# Add more steps...

# Run pipeline
results = framework.run_pipeline(pipeline, initial_inputs={"data_path": "data.csv", "config": config})


import os
import yaml
import logging
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Callable
import threading
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for the MLOps framework"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
    
    @staticmethod
    def save_config(config: Dict, config_path: str) -> None:
        """Save configuration to YAML file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as file:
                yaml.dump(config, file)
            logger.info(f"Config saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            raise

class ArtifactStore:
    """Store for managing pipeline artifacts"""
    
    def __init__(self, base_dir: str = "artifacts"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    def save_artifact(self, artifact: Any, name: str, metadata: Dict = None) -> str:
        """Save an artifact to the store"""
        try:
            # Create unique artifact ID
            artifact_id = str(uuid.uuid4())
            artifact_dir = os.path.join(self.base_dir, artifact_id)
            os.makedirs(artifact_dir, exist_ok=True)
            
            # Save metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "name": name,
                "created_at": datetime.now().isoformat(),
                "artifact_id": artifact_id
            })
            
            with open(os.path.join(artifact_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f)
            
            # Save the artifact
            artifact_path = os.path.join(artifact_dir, "artifact.pkl")
            
            # Handle different types of artifacts
            if isinstance(artifact, pd.DataFrame):
                artifact.to_pickle(artifact_path)
            else:
                import pickle
                with open(artifact_path, 'wb') as f:
                    pickle.dump(artifact, f)
            
            logger.info(f"Artifact {name} saved with ID {artifact_id}")
            return artifact_id
            
        except Exception as e:
            logger.error(f"Error saving artifact {name}: {e}")
            raise
    
    def load_artifact(self, artifact_id: str) -> Tuple[Any, Dict]:
        """Load an artifact from the store"""
        try:
            artifact_dir = os.path.join(self.base_dir, artifact_id)
            
            # Load metadata
            with open(os.path.join(artifact_dir, "metadata.json"), 'r') as f:
                metadata = json.load(f)
            
            # Load the artifact
            artifact_path = os.path.join(artifact_dir, "artifact.pkl")
            
            if not os.path.exists(artifact_path):
                raise FileNotFoundError(f"Artifact file not found: {artifact_path}")
            
            import pickle
            with open(artifact_path, 'rb') as f:
                artifact = pickle.load(f)
            
            logger.info(f"Artifact {metadata['name']} loaded from ID {artifact_id}")
            return artifact, metadata
            
        except Exception as e:
            logger.error(f"Error loading artifact {artifact_id}: {e}")
            raise
    
    def list_artifacts(self, filter_func: Callable = None) -> List[Dict]:
        """List artifacts in the store, optionally filtered"""
        try:
            artifacts = []
            
            for artifact_id in os.listdir(self.base_dir):
                artifact_dir = os.path.join(self.base_dir, artifact_id)
                if os.path.isdir(artifact_dir):
                    metadata_path = os.path.join(artifact_dir, "metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        if filter_func is None or filter_func(metadata):
                            artifacts.append(metadata)
            
            return artifacts
            
        except Exception as e:
            logger.error(f"Error listing artifacts: {e}")
            raise

class Step:
    """Base class for pipeline steps"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.step_id = str(uuid.uuid4())
        self.inputs = {}
        self.outputs = {}
        self.metadata = {}
    
    def execute(self, **kwargs):
        """Execute the step"""
        self.start_time = time.time()
        self.inputs = kwargs
        
        try:
            logger.info(f"Starting step: {self.name}")
            result = self._execute(**kwargs)
            self.outputs = result if isinstance(result, dict) else {"result": result}
            
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.metadata["duration"] = duration
            logger.info(f"Step {self.name} completed in {duration:.2f} seconds")
            
            return self.outputs
        except Exception as e:
            self.end_time = time.time()
            self.metadata["error"] = str(e)
            logger.error(f"Error in step {self.name}: {e}")
            raise
    
    def _execute(self, **kwargs):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")

class DataIngestionStep(Step):
    """Step for data ingestion"""
    
    def __init__(self, name: str = "data_ingestion"):
        super().__init__(name)
    
    def _execute(self, data_path: str, config: Dict) -> Dict:
        """Load and split data into training and test sets"""
        try:
            logger.info(f"Loading data from {data_path}")
            
            data = pd.read_csv(data_path)
            
            # Split data according to config
            train_size = config.get('data_split', {}).get('train_size', 0.8)
            target_column = config.get('data', {}).get('target_column')
            
            if not target_column:
                raise ValueError("Target column not specified in config")
                
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=42
            )
            
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
            
            logger.info(f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
            
            return {
                "train_data": train_data,
                "test_data": test_data
            }
            
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise

class FeatureEngineeringStep(Step):
    """Step for feature engineering"""
    
    def __init__(self, name: str = "feature_engineering"):
        super().__init__(name)
    
    def _execute(self, data: pd.DataFrame, config: Dict) -> Dict:
        """Process features according to configuration"""
        try:
            logger.info("Starting feature engineering")
            
            # Apply feature transformations based on config
            feature_config = config.get('features', {})
            processed_data = data.copy()
            
            # Handle categorical features
            categorical_features = feature_config.get('categorical_features', [])
            if categorical_features:
                processed_data = pd.get_dummies(processed_data, columns=categorical_features)
            
            # Handle numerical features
            numerical_features = feature_config.get('numerical_features', [])
            if numerical_features and feature_config.get('scale_numerical', False):
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                processed_data[numerical_features] = scaler.fit_transform(processed_data[numerical_features])
                
            logger.info(f"Feature engineering complete. Output shape: {processed_data.shape}")
            return {"processed_data": processed_data}
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {e}")
            raise

class ModelTrainingStep(Step):
    """Step for model training"""
    
    def __init__(self, name: str = "model_training", tracking_uri: str = None):
        super().__init__(name)
        self.tracking_uri = tracking_uri
        
        # Set up experiment tracking if URI provided
        if tracking_uri:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
    
    def _execute(self, train_data: pd.DataFrame, config: Dict) -> Dict:
        """Train machine learning model"""
        try:
            logger.info("Starting model training")
            
            # Extract model parameters from config
            model_config = config.get('model', {})
            model_type = model_config.get('type', 'random_forest')
            target_column = config.get('data', {}).get('target_column')
            
            if not target_column:
                raise ValueError("Target column not specified in config")
            
            X = train_data.drop(columns=[target_column])
            y = train_data[target_column]
            
            # Initialize model based on type
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                n_estimators = model_config.get('n_estimators', 100)
                max_depth = model_config.get('max_depth', None)
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            elif model_type == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingClassifier
                n_estimators = model_config.get('n_estimators', 100)
                learning_rate = model_config.get('learning_rate', 0.1)
                
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=42
                )
            elif model_type == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                C = model_config.get('C', 1.0)
                
                model = LogisticRegression(
                    C=C,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Track experiment if URI is set
            if self.tracking_uri:
                import mlflow
                import mlflow.sklearn
                
                experiment_name = config.get('experiment', {}).get('name', 'default')
                mlflow.set_experiment(experiment_name)
                
                with mlflow.start_run(run_name=f"{model_type}_training"):
                    # Log parameters
                    for param, value in model_config.items():
                        mlflow.log_param(param, value)
                    
                    # Train model
                    model.fit(X, y)
                    
                    # Log model
                    mlflow.sklearn.log_model(model, "model")
            else:
                # Train model without tracking
                model.fit(X, y)
            
            logger.info("Model training complete")
            return {"model": model}
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            if self.tracking_uri:
                import mlflow
                mlflow.end_run()
            raise

class ModelEvaluationStep(Step):
    """Step for model evaluation"""
    
    def __init__(self, name: str = "model_evaluation", tracking_uri: str = None):
        super().__init__(name)
        self.tracking_uri = tracking_uri
    
    def _execute(self, model: Any, test_data: pd.DataFrame, config: Dict) -> Dict:
        """Evaluate trained model on test data"""
        try:
            logger.info("Starting model evaluation")
            
            target_column = config.get('data', {}).get('target_column')
            
            if not target_column:
                raise ValueError("Target column not specified in config")
            
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Track metrics if URI is set
            if self.tracking_uri:
                import mlflow
                
                with mlflow.start_run(run_name="model_evaluation"):
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
            
            logger.info(f"Model evaluation complete. Metrics: {metrics}")
            return {"metrics": metrics}
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            if self.tracking_uri:
                import mlflow
                mlflow.end_run()
            raise

class ModelDeploymentStep(Step):
    """Step for model deployment"""
    
    def __init__(self, name: str = "model_deployment"):
        super().__init__(name)
        self.flask_app = None
        self.server_thread = None
    
    def _execute(self, model: Any, config: Dict) -> Dict:
        """Deploy trained model with Flask"""
        try:
            logger.info("Starting model deployment")
            
            from flask import Flask, request, jsonify
            from prometheus_client import start_http_server, Summary, Counter, Gauge
            
            # Set up Flask app
            app = Flask(__name__)
            self.flask_app = app
            
            # Set up Prometheus metrics
            prediction_counter = Counter('predictions_total', 'Total number of predictions')
            prediction_latency = Summary('prediction_latency_seconds', 'Prediction latency in seconds')
            model_version = config.get('model', {}).get('version', 'unknown')
            
            # Start Prometheus metrics server
            metrics_port = config.get('monitoring', {}).get('prometheus_port', 9090)
            start_http_server(metrics_port)
            
            # Register routes
            @app.route('/health', methods=['GET'])
            def health_check():
                return jsonify({'status': 'healthy', 'model_version': model_version})
            
            @app.route('/predict', methods=['POST'])
            def predict():
                try:
                    start_time = time.time()
                    
                    # Get data from request
                    data = request.json
                    
                    # Convert to DataFrame
                    input_df = pd.DataFrame([data])
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    
                    # Record prediction for monitoring
                    prediction_counter.inc()
                    prediction_latency.observe(time.time() - start_time)
                    
                    return jsonify({
                        'prediction': prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
                        'model_version': model_version
                    })
                    
                except Exception as e:
                    logger.error(f"Error during prediction: {e}")
                    return jsonify({'error': str(e)}), 500
            
            # Start Flask server in a separate thread
            flask_port = config.get('deployment', {}).get('flask_port', 5000)
            
            def run_flask():
                app.run(host='0.0.0.0', port=flask_port, debug=False, use_reloader=False)
            
            server_thread = threading.Thread(target=run_flask)
            server_thread.daemon = True
            server_thread.start()
            self.server_thread = server_thread
            
            logger.info(f"Model deployed successfully at http://localhost:{flask_port}")
            return {
                "status": "success",
                "endpoint": f"http://localhost:{flask_port}",
                "metrics_endpoint": f"http://localhost:{metrics_port}"
            }
            
        except Exception as e:
            logger.error(f"Error during model deployment: {e}")
            raise
    
    def shutdown(self):
        """Shutdown the deployment server"""
        if self.flask_app:
            import requests
            try:
                requests.get('http://localhost:5000/shutdown')
            except:
                pass

class ModelMonitoringStep(Step):
    """Step for model monitoring"""
    
    def __init__(self, name: str = "model_monitoring"):
        super().__init__(name)
    
    def _execute(self, reference_data: pd.DataFrame, config: Dict) -> Dict:
        """Set up model monitoring"""
        try:
            logger.info("Setting up model monitoring")
            
            # Create monitoring dashboard directory
            dashboard_path = os.path.join(
                config.get('monitoring', {}).get('dashboard_path', 'monitoring'),
                'dashboard.html'
            )
            os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
            
            # Set up Evidently dashboard if available
            try:
                from evidently.dashboard import Dashboard
                from evidently.dashboard.tabs import DataDriftTab, ModelPerformanceTab
                from evidently.pipeline.column_mapping import ColumnMapping
                
                target_column = config.get('data', {}).get('target_column')
                numerical_features = config.get('features', {}).get('numerical_features', [])
                categorical_features = config.get('features', {}).get('categorical_features', [])
                
                # Set up column mapping
                column_mapping = ColumnMapping(
                    target=target_column,
                    prediction='prediction',
                    numerical_features=numerical_features,
                    categorical_features=categorical_features
                )
                
                # Create dashboard
                dashboard = Dashboard(tabs=[
                    DataDriftTab(),
                    ModelPerformanceTab()
                ])
                
                # Save initial dashboard with reference data only
                dashboard.save(dashboard_path)
                
                logger.info(f"Monitoring dashboard created at {dashboard_path}")
                
                return {
                    "status": "success",
                    "dashboard_path": dashboard_path,
                    "column_mapping": column_mapping
                }
                
            except ImportError:
                logger.warning("Evidently not available. Monitoring dashboard not created.")
                return {
                    "status": "warning",
                    "message": "Evidently not available for monitoring"
                }
            
        except Exception as e:
            logger.error(f"Error during monitoring setup: {e}")
            raise

class ModelRetrainingStep(Step):
    """Step for model retraining"""
    
    def __init__(self, name: str = "model_retraining"):
        super().__init__(name)
    
    def _execute(self, metrics: Dict, config: Dict) -> Dict:
        """Check if retraining is needed based on metrics"""
        try:
            logger.info("Checking retraining trigger")
            
            retraining_config = config.get('retraining', {})
            metric_threshold = retraining_config.get('metric_threshold', {})
            
            retraining_needed = False
            triggered_by = []
            
            # Check if any metric falls below threshold
            for metric_name, threshold in metric_threshold.items():
                if metric_name in metrics and metrics[metric_name] < threshold:
                    retraining_needed = True
                    triggered_by.append(f"{metric_name}={metrics[metric_name]} below threshold {threshold}")
            
            if retraining_needed:
                logger.info(f"Retraining triggered: {', '.join(triggered_by)}")
                return {
                    "retraining_needed": True,
                    "triggered_by": triggered_by
                }
            else:
                logger.info("No retraining needed at this time")
                return {
                    "retraining_needed": False
                }
            
        except Exception as e:
            logger.error(f"Error during retraining check: {e}")
            raise

class Pipeline:
    """Pipeline for orchestrating MLOps steps"""
    
    def __init__(self, name: str):
        self.name = name
        self.steps = []
        self.artifact_store = ArtifactStore()
        self.execution_graph = {}
        self.results = {}
        self.pipeline_id = str(uuid.uuid4())
    
    def add_step(self, step: Step, dependencies: List[str] = None) -> None:
        """Add a step to the pipeline with optional dependencies"""
        self.steps.append(step)
        
        if dependencies:
            self.execution_graph[step.name] = dependencies
    
    def run(self, initial_inputs: Dict = None) -> Dict:
        """Run the pipeline"""
        try:
            logger.info(f"Starting pipeline: {self.name}")
            start_time = time.time()
            
            # Initialize results with initial inputs
            self.results = initial_inputs or {}
            
            # Track step dependencies and completed steps
            pending_steps = self.steps.copy()
            completed_steps = set()
            
            # Process steps
            while pending_steps:
                next_steps = []
                
                for step in pending_steps:
                    # Check if dependencies are met
                    if step.name in self.execution_graph:
                        dependencies = self.execution_graph[step.name]
                        if not all(dep in completed_steps for dep in dependencies):
                            next_steps.append(step)
                            continue
                    
                    # Execute step
                    logger.info(f"Executing step: {step.name}")
                    step_inputs = {}
                    
                    # Gather inputs for the step
                    for key, value in self.results.items():
                        step_inputs[key] = value
                    
                    # Execute the step
                    step_results = step.execute(**step_inputs)
                    
                    # Store artifacts
                    for key, value in step_results.items():
                        artifact_id = self.artifact_store.save_artifact(
                            value,
                            f"{step.name}_{key}",
                            {"step_id": step.step_id, "pipeline_id": self.pipeline_id}
                        )
                        
                        # Update results with artifacts
                        self.results[key] = value
                    
                    # Mark step as completed
                    completed_steps.add(step.name)
                
                # Update pending steps
                pending_steps = next_steps
                
                # Check for circular dependencies
                if len(pending_steps) == len(next_steps) and next_steps:
                    raise RuntimeError(f"Circular dependency detected among steps: {[s.name for s in next_steps]}")
            
            # Calculate execution time
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"Pipeline {self.name} completed in {duration:.2f} seconds")
            return self.results
            
        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise

class Executor:
    """Executor for running pipelines"""
    
    def __init__(self, mode: str = "local"):
        self.mode = mode
    
    def execute(self, pipeline: Pipeline, initial_inputs: Dict = None) -> Dict:
        """Execute a pipeline"""
        if self.mode == "local":
            return pipeline.run(initial_inputs)
        elif self.mode == "docker":
            # Would implement Docker-based execution here
            raise NotImplementedError("Docker execution not implemented yet")
        elif self.mode == "cloud":
            # Would implement cloud-based execution here
            raise NotImplementedError("Cloud execution not implemented yet")
        else:
            raise ValueError(f"Unsupported execution mode: {self.mode}")

class MLOpsFramework:
    """Main MLOps framework class"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = None
        
        if config_path:
            self.config = ConfigManager.load_config(config_path)
        
        self.artifact_store = ArtifactStore()
    
    def create_pipeline(self, name: str) -> Pipeline:
        """Create a new pipeline"""
        return Pipeline(name)
    
    def create_executor(self, mode: str = "local") -> Executor:
        """Create an executor"""
        return Executor(mode)
    
    def run_pipeline(self, pipeline: Pipeline, executor: Executor = None, initial_inputs: Dict = None) -> Dict:
        """Run a pipeline with the specified executor"""
        if executor is None:
            executor = Executor()
        
        return executor.execute(pipeline, initial_inputs)
    
    # Helper methods to create common steps
    def data_ingestion_step(self, name: str = "data_ingestion") -> DataIngestionStep:
        """Create a data ingestion step"""
        return DataIngestionStep(name)
    
    def feature_engineering_step(self, name: str = "feature_engineering") -> FeatureEngineeringStep:
        """Create a feature engineering step"""
        return FeatureEngineeringStep(name)
    
    def model_training_step(self, name: str = "model_training", tracking_uri: str = None) -> ModelTrainingStep:
        """Create a model training step"""
        return ModelTrainingStep(name, tracking_uri)
    
    def model_evaluation_step(self, name: str = "model_evaluation", tracking_uri: str = None) -> ModelEvaluationStep:
        """Create a model evaluation step"""
        return ModelEvaluationStep(name, tracking_uri)
    
    def model_deployment_step(self, name: str = "model_deployment") -> ModelDeploymentStep:
        """Create a model deployment step"""
        return ModelDeploymentStep(name)
    
    def model_monitoring_step(self, name: str = "model_monitoring") -> ModelMonitoringStep:
        """Create a model monitoring step"""
        return ModelMonitoringStep(name)
    
    def model_retraining_step(self, name: str = "model_retraining") -> ModelRetrainingStep:
        """Create a model retraining step"""
        return ModelRetrainingStep(name)

# Usage example
def main():
    """Example usage of the MLOps framework"""
    # Initialize the framework
    framework = MLOpsFramework("config/config.yml")
    
    # Create a pipeline
    pipeline = framework.create_pipeline("example_pipeline")
    
    # Add steps to the pipeline
    data_step = framework.data_ingestion_step()
    pipeline.add_step(data_step)
    
    feature_step = framework.feature_engineering_step()
    pipeline.add_step(feature_step, dependencies=["data_ingestion"])
    
    training_step = framework.model_training_step(tracking_uri="mlflow:///tmp/mlflow")
    pipeline.add_step(training_step, dependencies=["feature_engineering"])
    
    eval_step = framework.model_evaluation_step(tracking_uri="mlflow:///tmp/mlflow")
    pipeline.add_step(eval_step, dependencies=["model_training"])
    
    deploy_step = framework.model_deployment_step()
    pipeline.add_step(deploy_step, dependencies=["model_training", "model_evaluation"])
    
    monitoring_step = framework.model_monitoring_step()
    pipeline.add_step(monitoring_step, dependencies=["model_deployment"])
    
    retraining_step = framework.model_retraining_step()
    pipeline.add_step(retraining_step, dependencies=["model_evaluation"])
    
    # Create an executor
    executor = framework.create_executor(mode="local")
    
    # Run the pipeline
    initial_inputs = {
        "data_path": "data/data.csv",
        "config": framework.config
    }
    
    results = framework.run_pipeline(pipeline, executor, initial_inputs)
    print(f"Pipeline Results: {results}")

if __name__ == "__main__":
    main()