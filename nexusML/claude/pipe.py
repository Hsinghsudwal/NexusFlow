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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """Configuration loader class"""
    
    @staticmethod
    def load_file(config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise

class DataIngestion:
    """Data ingestion component"""
    
    @task
    def data_ingestion(self, path: str, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split data into training and test sets"""
        logger.info(f"Loading data from {path}")
        
        try:
            data = pd.read_csv(path)
            
            # Split data according to config
            train_size = config.get('data_split', {}).get('train_size', 0.8)
            target_column = config.get('data', {}).get('target_column')
            
            if not target_column:
                raise ValueError("Target column not specified in config")
                
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=42
            )
            
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
            
            logger.info(f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
            
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
            feature_config = config.get('features', {})
            processed_data = data.copy()
            
            # Handle categorical features
            categorical_features = feature_config.get('categorical_features', [])
            if categorical_features:
                processed_data = pd.get_dummies(processed_data, columns=categorical_features)
            
            # Handle numerical features (example: scaling)
            # This could be expanded with more transformations
                
            logger.info(f"Feature engineering complete. Output shape: {processed_data.shape}")
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
            model_config = config.get('model', {})
            model_type = model_config.get('type', 'random_forest')
            target_column = config.get('data', {}).get('target_column')
            
            X = train_data.drop(columns=[target_column])
            y = train_data[target_column]
            
            # Initialize model based on type
            if model_type == 'random_forest':
                n_estimators = model_config.get('n_estimators', 100)
                max_depth = model_config.get('max_depth', None)
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Start MLflow run
            mlflow.start_run(run_name=config.get('experiment_name', 'model_training'))
            
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
            target_column = config.get('data', {}).get('target_column')
            
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Log metrics with MLflow
            mlflow.start_run(run_name=config.get('experiment_name', 'model_evaluation'))
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.end_run()
            
            logger.info(f"Model evaluation complete. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            mlflow.end_run()
            raise

class ModelDeployment:
    """Model deployment component using Flask"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.model = None
        self.config = None
        
    def setup_routes(self):
        """Set up Flask routes for prediction"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({'status': 'healthy'})
        
        @self.app.route('/predict', methods=['POST'])
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
                
                return jsonify({
                    'prediction': prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
                    'model_version': self.config.get('model', {}).get('version', 'unknown')
                })
                
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                return jsonify({'error': str(e)}), 500
    
    @task
    def deploy_model(self, model: Any, config: Dict) -> bool:
        """Deploy trained model with Flask"""
        logger.info("Starting model deployment")
        
        try:
            self.model = model
            self.config = config
            
            # Set up Prometheus metrics
            self.prediction_counter = Counter('predictions_total', 'Total number of predictions')
            self.prediction_latency = Summary('prediction_latency_seconds', 'Prediction latency in seconds')
            
            # Start Prometheus metrics server
            metrics_port = config.get('monitoring', {}).get('prometheus_port', 9090)
            start_http_server(metrics_port)
            
            # Setup Flask routes
            self.setup_routes()
            
            # Start Flask server in a separate thread
            flask_port = config.get('deployment', {}).get('flask_port', 5000)
            from threading import Thread
            server_thread = Thread(target=lambda: self.app.run(
                host='0.0.0.0', 
                port=flask_port,
                debug=False,
                use_reloader=False
            ))
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
            
            target_column = config.get('data', {}).get('target_column')
            numerical_features = config.get('features', {}).get('numerical_features', [])
            categorical_features = config.get('features', {}).get('categorical_features', [])
            
            # Set up column mapping for Evidently
            self.column_mapping = ColumnMapping(
                target=target_column,
                prediction='prediction',
                numerical_features=numerical_features,
                categorical_features=categorical_features
            )
            
            # Create Evidently dashboard
            self.dashboard = Dashboard(tabs=[
                DataDriftTab(),
                ModelPerformanceTab()
            ])
            
            # Create dashboard output directory
            dashboard_path = os.path.join(
                config.get('monitoring', {}).get('dashboard_path', 'monitoring'),
                'dashboard.html'
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
        data_record['prediction'] = prediction
        self.current_data.append(data_record)
    
    def update_dashboard(self, dashboard_path: str) -> None:
        """Update monitoring dashboard with new data"""
        if len(self.current_data) > 0:
            current_df = pd.DataFrame(self.current_data)
            self.dashboard.calculate(self.reference_data, current_df, column_mapping=self.column_mapping)
            self.dashboard.save(dashboard_path)

class ModelRetraining:
    """Model retraining component"""
    
    @task
    def check_retraining_trigger(self, metrics: Dict, config: Dict) -> bool:
        """Check if retraining is needed based on metrics"""
        logger.info("Checking retraining trigger")
        
        try:
            retraining_config = config.get('retraining', {})
            metric_threshold = retraining_config.get('metric_threshold', {})
            
            # Check if any metric falls below threshold
            for metric_name, threshold in metric_threshold.items():
                if metric_name in metrics and metrics[metric_name] < threshold:
                    logger.info(f"Retraining triggered: {metric_name}={metrics[metric_name]} below threshold {threshold}")
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
    
    def __init__(self, name: str, max_workers: int = 4):
        self.name = name
        self.max_workers = max_workers
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
        processed_test_data = self.feature_engineering.process_features(test_data, config)
        metrics = self.model_evaluation.evaluate_model(model, processed_test_data, config)
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
        self.config = Config.load_file('config/config.yml')
    
    @flow(name="MLOps Pipeline", task_runner=ConcurrentTaskRunner())
    def run(self) -> Dict:
        """Run the complete MLOps pipeline"""
        logger.info(f"Starting MLOps pipeline for data at {self.path}")
        
        try:
            # Initialize stack
            option_run= LOCAL/CLOUD
            option_scale=LOCAL/DOCKER

            my_stack = Stack("ml_pipeline", max_workers=4, running=option_run,scaler = option_scale)
            
            # Data ingestion
            train_data, test_data = my_stack.data_ingestion.data_ingestion(self.path, self.config)
            
            # Model training
            model = my_stack.train(train_data, self.config)
            
            # Model evaluation
            metrics = my_stack.evaluate(model, test_data, self.config)
            
            # Model deployment
            deployment_success = my_stack.deploy(model, self.config)
            
            # Set up monitoring
            dashboard_path = my_stack.monitor(test_data, self.config)
            
            # Check if retraining is needed
            retraining_success = my_stack.check_and_retrain(metrics, self.path, self.config)
            
            # Return results
            results = {
                'deployment': deployment_success,
                'monitoring': dashboard_path,
                're-training': retraining_success
            }
            
            logger.info(f"MLOps pipeline completed successfully. Results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise

def main():
    """Main entry point"""
    # Create and run pipeline
    path = 'data/data.csv'
    pipeline = TrainingPipeline(path)
    results = pipeline.run()
    print(f"Pipeline Results: {results}")

if __name__ == '__main__':
    main()