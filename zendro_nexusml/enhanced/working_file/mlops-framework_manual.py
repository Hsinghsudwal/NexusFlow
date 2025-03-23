# MLOps Framework with Integrated Pipeline Structure
# Core components for a complete ML lifecycle

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# ============= CONFIG MANAGEMENT =============
class MLConfig:
    """Configuration manager for ML projects"""
    
    def __init__(self, config_path: str = None):
        self.config = {}
        self.config_path = config_path
        if config_path and os.path.exists(config_path):
            self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        import yaml
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def save_config(self, path: str = None):
        """Save configuration to file"""
        import yaml
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set config value"""
        self.config[key] = value

# ============= DATA MANAGEMENT =============
class DataManager:
    """Handles data versioning, processing and storage"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger("DataManager")
    
    def load_data(self, source: str, **kwargs) -> Any:
        """Load data from various sources"""
        if source.endswith('.csv'):
            import pandas as pd
            return pd.read_csv(source, **kwargs)
        elif source.endswith(('.parquet', '.pq')):
            import pandas as pd
            return pd.read_parquet(source, **kwargs)
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    def version_dataset(self, data: Any, name: str) -> str:
        """Version a dataset and store metadata"""
        import hashlib
        import json
        
        # Create a simple hash of the data
        data_hash = hashlib.md5(str(data).encode()).hexdigest()
        
        # Create version metadata
        version_info = {
            "name": name,
            "hash": data_hash,
            "timestamp": datetime.now().isoformat(),
            "rows": len(data) if hasattr(data, "__len__") else "unknown"
        }
        
        # Store version info
        version_path = os.path.join(
            self.config.get("data_registry", "data_versions"),
            f"{name}_{data_hash[:8]}.json"
        )
        os.makedirs(os.path.dirname(version_path), exist_ok=True)
        
        with open(version_path, 'w') as f:
            json.dump(version_info, f)
        
        self.logger.info(f"Versioned dataset {name} with hash {data_hash[:8]}")
        return data_hash[:8]
    
    def process_data(self, data: Any, pipeline: List[callable]) -> Any:
        """Apply a sequence of processing steps to data"""
        result = data
        for step in pipeline:
            result = step(result)
        return result

# ============= MODEL MANAGEMENT =============
class ModelManager:
    """Manages model training, versioning and deployment"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger("ModelManager")
        self.model_registry = self.config.get("model_registry", "models")
        os.makedirs(self.model_registry, exist_ok=True)
    
    def train_model(self, model, X_train, y_train, **kwargs):
        """Train a model with the given data"""
        self.logger.info(f"Training model: {type(model).__name__}")
        model.fit(X_train, y_train, **kwargs)
        return model
    
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        """Evaluate model performance"""
        from sklearn import metrics
        
        self.logger.info(f"Evaluating model: {type(model).__name__}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        eval_metrics = {}
        
        # For classification
        if len(set(y_test)) < 10:  # Simple heuristic to detect classification
            eval_metrics["accuracy"] = metrics.accuracy_score(y_test, y_pred)
            eval_metrics["f1_score"] = metrics.f1_score(y_test, y_pred, average='weighted')
            
            try:
                y_proba = model.predict_proba(X_test)
                eval_metrics["auc_roc"] = metrics.roc_auc_score(y_test, y_proba[:, 1])
            except (AttributeError, IndexError):
                pass
        
        # For regression
        else:
            eval_metrics["mse"] = metrics.mean_squared_error(y_test, y_pred)
            eval_metrics["rmse"] = eval_metrics["mse"] ** 0.5
            eval_metrics["mae"] = metrics.mean_absolute_error(y_test, y_pred)
            eval_metrics["r2"] = metrics.r2_score(y_test, y_pred)
        
        return eval_metrics
    
    def save_model(self, model, name: str, metadata: Dict = None) -> str:
        """Save model to the model registry with versioning"""
        import joblib
        import json
        import hashlib
        
        # Generate model version
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_id = f"{name}_{timestamp}"
        
        # Save model file
        model_path = os.path.join(self.model_registry, f"{model_id}.joblib")
        joblib.dump(model, model_path)
        
        # Calculate model file hash
        with open(model_path, 'rb') as f:
            model_hash = hashlib.md5(f.read()).hexdigest()
        
        # Save metadata
        model_meta = metadata or {}
        model_meta.update({
            "name": name,
            "id": model_id,
            "hash": model_hash,
            "timestamp": datetime.now().isoformat(),
            "type": type(model).__name__
        })
        
        meta_path = os.path.join(self.model_registry, f"{model_id}.json")
        with open(meta_path, 'w') as f:
            json.dump(model_meta, f)
        
        self.logger.info(f"Saved model {name} with ID {model_id}")
        return model_id
    
    def load_model(self, model_id: str):
        """Load a model from the model registry"""
        import joblib
        
        model_path = os.path.join(self.model_registry, f"{model_id}.joblib")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_id} not found in registry")
        
        model = joblib.load(model_path)
        self.logger.info(f"Loaded model {model_id}")
        return model

# ============= EXPERIMENT TRACKING =============
class ExperimentTracker:
    """Tracks experiment runs and results"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger("ExperimentTracker")
        self.experiment_dir = self.config.get("experiment_dir", "experiments")
        os.makedirs(self.experiment_dir, exist_ok=True)
    
    def start_run(self, experiment_name: str, run_params: Dict = None) -> str:
        """Start a new experiment run"""
        run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        run_info = {
            "experiment": experiment_name,
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "params": run_params or {}
        }
        
        # Save run info
        run_path = os.path.join(self.experiment_dir, f"{run_id}.json")
        self._save_run_info(run_info, run_path)
        
        self.logger.info(f"Started experiment run {run_id}")
        return run_id
    
    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: int = None):
        """Log metrics for a run"""
        run_path = os.path.join(self.experiment_dir, f"{run_id}.json")
        if not os.path.exists(run_path):
            raise ValueError(f"Run {run_id} not found")
        
        run_info = self._load_run_info(run_path)
        
        # Initialize metrics if not present
        if "metrics" not in run_info:
            run_info["metrics"] = {}
        
        # Add step dimension if provided
        if step is not None:
            for key, value in metrics.items():
                if key not in run_info["metrics"]:
                    run_info["metrics"][key] = []
                run_info["metrics"][key].append({"step": step, "value": value})
        else:
            # Just update with latest metrics
            run_info["metrics"].update(metrics)
        
        self._save_run_info(run_info, run_path)
    
    def end_run(self, run_id: str, status: str = "completed"):
        """End an experiment run"""
        run_path = os.path.join(self.experiment_dir, f"{run_id}.json")
        if not os.path.exists(run_path):
            raise ValueError(f"Run {run_id} not found")
        
        run_info = self._load_run_info(run_path)
        run_info["end_time"] = datetime.now().isoformat()
        run_info["status"] = status
        
        self._save_run_info(run_info, run_path)
        self.logger.info(f"Ended experiment run {run_id} with status: {status}")
    
    def _load_run_info(self, path: str) -> Dict:
        """Load run info from file"""
        import json
        with open(path, 'r') as f:
            return json.load(f)
    
    def _save_run_info(self, run_info: Dict, path: str):
        """Save run info to file"""
        import json
        with open(path, 'w') as f:
            json.dump(run_info, f, indent=2)

# ============= DEPLOYMENT MANAGER =============
class DeploymentManager:
    """Manages model deployment and serving"""
    
    def __init__(self, config: MLConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.logger = logging.getLogger("DeploymentManager")
        self.deployment_dir = self.config.get("deployment_dir", "deployments")
        os.makedirs(self.deployment_dir, exist_ok=True)
    
    def deploy_model(self, model_id: str, endpoint_name: str) -> Dict:
        """Deploy a model to a serving endpoint"""
        # Load model metadata
        import json
        meta_path = os.path.join(self.model_manager.model_registry, f"{model_id}.json")
        
        if not os.path.exists(meta_path):
            raise ValueError(f"Model {model_id} not found in registry")
        
        with open(meta_path, 'r') as f:
            model_meta = json.load(f)
        
        # Create deployment record
        deployment = {
            "endpoint_name": endpoint_name,
            "model_id": model_id,
            "model_name": model_meta.get("name"),
            "deployment_time": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Save deployment record
        deploy_path = os.path.join(self.deployment_dir, f"{endpoint_name}.json")
        with open(deploy_path, 'w') as f:
            json.dump(deployment, f, indent=2)
        
        self.logger.info(f"Deployed model {model_id} to endpoint {endpoint_name}")
        
        # Create simple prediction function wrapper
        def predict_fn(data):
            model = self.model_manager.load_model(model_id)
            return model.predict(data)
        
        return {"deployment": deployment, "predict": predict_fn}
    
    def list_deployments(self) -> List[Dict]:
        """List all active deployments"""
        import json
        
        deployments = []
        for filename in os.listdir(self.deployment_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.deployment_dir, filename), 'r') as f:
                    deployment = json.load(f)
                    deployments.append(deployment)
        
        return deployments
    
    def get_deployment(self, endpoint_name: str) -> Dict:
        """Get deployment information for an endpoint"""
        import json
        
        deploy_path = os.path.join(self.deployment_dir, f"{endpoint_name}.json")
        if not os.path.exists(deploy_path):
            raise ValueError(f"Endpoint {endpoint_name} not found")
        
        with open(deploy_path, 'r') as f:
            return json.load(f)

# ============= MONITORING =============
class ModelMonitor:
    """Monitors model performance in production"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger("ModelMonitor")
        self.monitoring_dir = self.config.get("monitoring_dir", "monitoring")
        os.makedirs(self.monitoring_dir, exist_ok=True)
    
    def log_prediction(self, endpoint_name: str, inputs: Any, prediction: Any, actual: Any = None):
        """Log a prediction for monitoring"""
        import json
        import numpy as np
        import pandas as pd
        
        # Convert numpy/pandas types to native Python types for JSON serialization
        def serialize(obj):
            if isinstance(obj, (np.ndarray, pd.Series)):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint_name,
            "inputs": serialize(inputs),
            "prediction": serialize(prediction)
        }
        
        if actual is not None:
            log_entry["actual"] = serialize(actual)
        
        # Append to log file
        log_path = os.path.join(self.monitoring_dir, f"{endpoint_name}_predictions.jsonl")
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def analyze_performance(self, endpoint_name: str, period_days: int = 7) -> Dict:
        """Analyze recent model performance for an endpoint"""
        import json
        from datetime import datetime, timedelta
        
        log_path = os.path.join(self.monitoring_dir, f"{endpoint_name}_predictions.jsonl")
        if not os.path.exists(log_path):
            return {"error": f"No logs found for endpoint {endpoint_name}"}
        
        # Set time threshold
        threshold = datetime.now() - timedelta(days=period_days)
        
        # Collect logs with actual values for evaluation
        logs_with_actual = []
        
        with open(log_path, 'r') as f:
            for line in f:
                log = json.loads(line)
                log_time = datetime.fromisoformat(log["timestamp"])
                
                if log_time >= threshold and "actual" in log:
                    logs_with_actual.append(log)
        
        if not logs_with_actual:
            return {"warning": f"No logs with actual values found for endpoint {endpoint_name} in the last {period_days} days"}
        
        # Calculate basic metrics
        from sklearn import metrics
        
        y_true = [log["actual"] for log in logs_with_actual]
        y_pred = [log["prediction"] for log in logs_with_actual]
        
        # Simple type check to determine metric type
        if isinstance(y_true[0], (list, int, float)):
            # For regression or binary classification
            results = {
                "count": len(y_true),
                "period_days": period_days
            }
            
            # Check if binary classification or regression
            unique_values = set()
            for val in y_true:
                if isinstance(val, list):
                    unique_values.update(val)
                else:
                    unique_values.add(val)
            
            if len(unique_values) <= 2:  # Binary classification
                results["accuracy"] = metrics.accuracy_score(y_true, y_pred)
                results["f1_score"] = metrics.f1_score(y_true, y_pred, average='binary')
            else:  # Regression
                results["mse"] = metrics.mean_squared_error(y_true, y_pred)
                results["rmse"] = results["mse"] ** 0.5
                results["mae"] = metrics.mean_absolute_error(y_true, y_pred)
        
        return results

# ============= PIPELINE =============
class MLPipeline:
    """Orchestrates the ML workflow pipeline"""
    
    def __init__(self, config_path: str = None):
        # Initialize the pipeline components
        self.config = MLConfig(config_path)
        self.data_manager = DataManager(self.config)
        self.model_manager = ModelManager(self.config)
        self.experiment_tracker = ExperimentTracker(self.config)
        self.deployment_manager = DeploymentManager(self.config, self.model_manager)
        self.model_monitor = ModelMonitor(self.config)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("MLPipeline")
    
    def run(self, pipeline_config: Dict) -> Dict:
        """Run the full ML pipeline based on config"""
        experiment_name = pipeline_config.get("experiment_name", f"experiment_{datetime.now().strftime('%Y%m%d')}")
        
        # Start experiment tracking
        run_id = self.experiment_tracker.start_run(experiment_name, pipeline_config)
        results = {"run_id": run_id}
        
        try:
            # 1. Data Loading
            self.logger.info("Step 1: Loading data")
            data_source = pipeline_config.get("data_source")
            if not data_source:
                raise ValueError("Data source not specified in pipeline configuration")
            
            data = self.data_manager.load_data(data_source)
            data_version = self.data_manager.version_dataset(data, "raw_data")
            results["data_version"] = data_version
            
            # 2. Data Processing
            self.logger.info("Step 2: Processing data")
            processing_steps = pipeline_config.get("processing_steps", [])
            
            # Convert string processing steps to actual functions if needed
            processed_data = self.data_manager.process_data(data, processing_steps)
            
            # 3. Train/Test Split
            self.logger.info("Step 3: Splitting data")
            from sklearn.model_selection import train_test_split
            
            target_col = pipeline_config.get("target_column")
            features = pipeline_config.get("feature_columns", processed_data.columns.tolist())
            
            if target_col in features:
                features.remove(target_col)
            
            X = processed_data[features]
            y = processed_data[target_col]
            
            test_size = pipeline_config.get("test_size", 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=pipeline_config.get("random_state", 42)
            )
            
            # Log train/test sizes
            self.experiment_tracker.log_metrics(run_id, {
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            })
            
            # 4. Model Training
            self.logger.info("Step 4: Training model")
            model_config = pipeline_config.get("model", {})
            model_type = model_config.get("type", "RandomForest")
            
            # Create model instance based on config
            if model_type == "RandomForest":
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                if pipeline_config.get("problem_type") == "regression":
                    model = RandomForestRegressor(**model_config.get("params", {}))
                else:
                    model = RandomForestClassifier(**model_config.get("params", {}))
            
            elif model_type == "LogisticRegression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(**model_config.get("params", {}))
            
            elif model_type == "LinearRegression":
                from sklearn.linear_model import LinearRegression
                model = LinearRegression(**model_config.get("params", {}))
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train the model
            trained_model = self.model_manager.train_model(model, X_train, y_train)
            
            # 5. Model Evaluation
            self.logger.info("Step 5: Evaluating model")
            evaluation = self.model_manager.evaluate_model(trained_model, X_test, y_test)
            results["evaluation"] = evaluation
            
            # Log evaluation metrics
            self.experiment_tracker.log_metrics(run_id, evaluation)
            
            # 6. Model Saving
            self.logger.info("Step 6: Saving model")
            model_name = pipeline_config.get("model_name", f"{model_type}_{datetime.now().strftime('%Y%m%d')}")
            
            model_metadata = {
                "features": features,
                "target": target_col,
                "evaluation": evaluation,
                "data_version": data_version,
                "experiment_run": run_id,
                **model_config
            }
            
            model_id = self.model_manager.save_model(trained_model, model_name, model_metadata)
            results["model_id"] = model_id
            
            # 7. Optional - Model Deployment
            if pipeline_config.get("deploy", False):
                self.logger.info("Step 7: Deploying model")
                endpoint_name = pipeline_config.get("endpoint_name", f"{model_name}_endpoint")
                
                deployment = self.deployment_manager.deploy_model(model_id, endpoint_name)
                results["deployment"] = deployment["deployment"]
            
            # End experiment run as completed
            self.experiment_tracker.end_run(run_id, "completed")
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            self.experiment_tracker.end_run(run_id, "failed")
            results["error"] = str(e)
        
        return results

# ============= EXAMPLE USAGE =============
def example_usage():
    """Example of how to use the MLOps framework"""
    
    # Define pipeline configuration
    pipeline_config = {
        "experiment_name": "customer_churn_prediction",
        "data_source": "data/customer_data.csv",
        "target_column": "churn",
        "problem_type": "classification",
        "test_size": 0.25,
        "random_state": 42,
        "model": {
            "type": "RandomForest",
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5
            }
        },
        "model_name": "churn_predictor",
        "deploy": True,
        "endpoint_name": "churn_prediction_api"
    }
    
    # Initialize and run the pipeline
    pipeline = MLPipeline()
    results = pipeline.run(pipeline_config)
    
    print(f"Pipeline completed. Model ID: {results.get('model_id')}")
    
    # After deployment, monitor the model
    if "deployment" in results:
        endpoint = results["deployment"]["endpoint_name"]
        
        # Example of logging predictions
        import numpy as np
        
        # Simulate some prediction data
        for i in range(10):
            # Fake input data
            inputs = np.random.rand(5)
            
            # Get model and make prediction
            model = pipeline.model_manager.load_model(results["model_id"])
            prediction = model.predict([inputs])[0]
            
            # Simulate actual value for monitoring
            actual = 1 if prediction > 0.5 else 0
            
            # Log the prediction
            pipeline.model_monitor.log_prediction(endpoint, inputs, prediction, actual)
        
        # Analyze performance after some time
        perf = pipeline.model_monitor.analyze_performance(endpoint, period_days=1)
        print(f"Model performance: {perf}")

if __name__ == "__main__":
    example_usage()
