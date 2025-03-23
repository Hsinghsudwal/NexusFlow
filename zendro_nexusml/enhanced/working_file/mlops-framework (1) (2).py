import os
import logging
import yaml
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mlops-framework")

class MLOpsConfig:
    """Configuration manager for the MLOps framework."""
    
    def __init__(self, config_path: str):
        """Initialize the configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self.config.get(key, default)


class DataIngestionStage:
    """Handles data ingestion from various sources."""
    
    def __init__(self, config: MLOpsConfig):
        """Initialize with configuration.
        
        Args:
            config: MLOps configuration instance
        """
        self.config = config
        self.data_sources = config.get("data_sources", [])
        
    def execute(self) -> Dict[str, str]:
        """Execute the data ingestion process.
        
        Returns:
            Dictionary with paths to ingested data
        """
        logger.info("Starting data ingestion")
        result = {}
        
        for source in self.data_sources:
            source_name = source.get("name")
            source_type = source.get("type")
            source_path = source.get("path")
            
            try:
                # In a real implementation, this would handle different source types
                # (e.g., S3, database, local files)
                output_path = f"data/raw/{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                logger.info(f"Ingested data from {source_name} to {output_path}")
                result[source_name] = output_path
            except Exception as e:
                logger.error(f"Failed to ingest data from {source_name}: {str(e)}")
                raise
                
        return result


class DataPreprocessingStage:
    """Handles data preprocessing and feature engineering."""
    
    def __init__(self, config: MLOpsConfig):
        """Initialize with configuration.
        
        Args:
            config: MLOps configuration instance
        """
        self.config = config
        self.preprocessing_steps = config.get("preprocessing_steps", [])
        
    def execute(self, input_data: Dict[str, str]) -> Dict[str, str]:
        """Execute preprocessing steps on input data.
        
        Args:
            input_data: Dictionary with paths to raw data
            
        Returns:
            Dictionary with paths to processed data
        """
        logger.info("Starting data preprocessing")
        result = {}
        
        for source_name, source_path in input_data.items():
            try:
                # In a real implementation, this would apply preprocessing steps
                output_path = f"data/processed/{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                logger.info(f"Preprocessed data from {source_name} to {output_path}")
                result[source_name] = output_path
            except Exception as e:
                logger.error(f"Failed to preprocess data from {source_name}: {str(e)}")
                raise
                
        return result


class ModelTrainingStage:
    """Handles model training and hyperparameter tuning."""
    
    def __init__(self, config: MLOpsConfig):
        """Initialize with configuration.
        
        Args:
            config: MLOps configuration instance
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.experiment_tracker = ExperimentTracker(config)
        self.retraining_config = config.get("retraining", {})
        self.current_attempt = 0
        self.max_attempts = self.retraining_config.get("max_attempts", 3)
        
    def execute(self, input_data: Dict[str, str], is_retraining: bool = False) -> str:
        """Train model with input data.
        
        Args:
            input_data: Dictionary with paths to processed data
            is_retraining: Flag indicating if this is a retraining run
            
        Returns:
            Path to the trained model
        """
        if is_retraining:
            self.current_attempt += 1
            logger.info(f"Starting model retraining (attempt {self.current_attempt}/{self.max_attempts})")
            
            # Apply retraining-specific configurations
            strategy = self.retraining_config.get("strategy", "default")
            if strategy == "adjust_hyperparams":
                self._adjust_hyperparameters()
            elif strategy == "more_data":
                # In a real system, this might fetch additional data
                pass
        else:
            logger.info("Starting model training")
            self.current_attempt = 0
        
        run_id = self.experiment_tracker.start_run()
        
        try:
            model_type = self.model_config.get("type", "default")
            hyperparams = self.model_config.get("hyperparameters", {})
            
            logger.info(f"Training {model_type} model with hyperparameters: {hyperparams}")
            
            # Mock training process
            model_path = f"models/{model_type}_{run_id}"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Simulate different performance for retraining attempts
            base_accuracy = 0.85
            if is_retraining:
                # Simulate improvement with retraining
                base_accuracy = min(0.95, base_accuracy + (self.current_attempt * 0.03))
            
            # Log metrics
            metrics = {
                "accuracy": base_accuracy,
                "precision": base_accuracy - 0.03,
                "recall": base_accuracy + 0.03,
                "f1_score": base_accuracy
            }
            
            self.experiment_tracker.log_metrics(metrics)
            self.experiment_tracker.log_parameter("is_retraining", is_retraining)
            self.experiment_tracker.log_parameter("retraining_attempt", self.current_attempt)
            
            self.experiment_tracker.end_run()
            logger.info(f"Model trained and saved to {model_path}")
            
            return model_path
        except Exception as e:
            self.experiment_tracker.log_failure(str(e))
            self.experiment_tracker.end_run()
            logger.error(f"Failed to train model: {str(e)}")
            raise
    
    def _adjust_hyperparameters(self) -> None:
        """Adjust hyperparameters based on retraining strategy."""
        # In a real system, this might use Bayesian optimization or other strategies
        current_hyperparams = self.model_config.get("hyperparameters", {})
        
        # Example strategy: increase model complexity with each attempt
        if "max_depth" in current_hyperparams:
            current_hyperparams["max_depth"] += 2
            
        if "learning_rate" in current_hyperparams:
            current_hyperparams["learning_rate"] *= 0.9
            
        # Update hyperparameters
        self.model_config["hyperparameters"] = current_hyperparams
        logger.info(f"Adjusted hyperparameters for retraining: {current_hyperparams}")


class ModelEvaluationStage:
    """Handles model evaluation and validation."""
    
    def __init__(self, config: MLOpsConfig):
        """Initialize with configuration.
        
        Args:
            config: MLOps configuration instance
        """
        self.config = config
        self.evaluation_metrics = config.get("evaluation_metrics", ["accuracy"])
        self.thresholds = config.get("evaluation_thresholds", {})
        self.experiment_tracker = ExperimentTracker(config)
        
    def execute(self, model_path: str, test_data: Dict[str, str]) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            model_path: Path to the trained model
            test_data: Dictionary with paths to test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model at {model_path}")
        
        # In a real implementation, this would:
        # 1. Load the model
        # 2. Run predictions on test data
        # 3. Calculate specified metrics
        # 4. Compare against thresholds
        
        run_id = self.experiment_tracker.start_run()
        
        try:
            # Extract model type and run id for simulating different performances
            model_info = os.path.basename(model_path).split('_')
            is_retraining = "retraining" in model_path
            
            # Mock evaluation process - in a real system this would actually evaluate the model
            if is_retraining:
                # Simulate improvement in retraining
                base_performance = 0.83 + (0.03 * model_path.count("retraining"))
                base_performance = min(0.93, base_performance)
            else:
                base_performance = 0.83
                
            metrics = {
                "accuracy": base_performance,
                "precision": base_performance - 0.03,
                "recall": base_performance + 0.02,
                "f1_score": base_performance - 0.01
            }
            
            self.experiment_tracker.log_metrics(metrics)
            self.experiment_tracker.end_run()
            
            # Check if metrics meet thresholds
            all_metrics_pass = True
            failing_metrics = {}
            
            for metric_name, threshold in self.thresholds.items():
                if metric_name in metrics and metrics[metric_name] < threshold:
                    all_metrics_pass = False
                    failing_metrics[metric_name] = {
                        "value": metrics[metric_name],
                        "threshold": threshold
                    }
                    logger.warning(f"Metric {metric_name} is below threshold: {metrics[metric_name]} < {threshold}")
            
            metrics["all_metrics_pass"] = all_metrics_pass
            metrics["failing_metrics"] = failing_metrics
            
            logger.info(f"Model evaluation complete: {metrics}")
            return metrics
        except Exception as e:
            self.experiment_tracker.log_failure(str(e))
            self.experiment_tracker.end_run()
            logger.error(f"Failed to evaluate model: {str(e)}")
            raise


class ModelDeploymentStage:
    """Handles model deployment and serving."""
    
    def __init__(self, config: MLOpsConfig):
        """Initialize with configuration.
        
        Args:
            config: MLOps configuration instance
        """
        self.config = config
        self.deployment_config = config.get("deployment", {})
        
    def execute(self, model_path: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Deploy the model if it meets criteria.
        
        Args:
            model_path: Path to the trained model
            metrics: Dictionary with evaluation metrics
            
        Returns:
            Dictionary with deployment information
        """
        logger.info(f"Preparing to deploy model from {model_path}")
        
        deployment_target = self.deployment_config.get("target", "local")
        deployment_thresholds = self.deployment_config.get("thresholds", {})
        
        # Check if model meets deployment criteria
        for metric_name, threshold in deployment_thresholds.items():
            if metric_name in metrics and metrics[metric_name] < threshold:
                logger.warning(f"Model does not meet deployment criteria. {metric_name}: {metrics[metric_name]} < {threshold}")
                return {
                    "status": "rejected", 
                    "reason": f"{metric_name} below threshold",
                    "metrics": metrics
                }
        
        try:
            # In a real implementation, this would:
            # 1. Package the model for deployment
            # 2. Deploy to the target environment (e.g., cloud service, edge device)
            # 3. Set up monitoring
            
            # Mock deployment process
            deployment_id = f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Model deployed successfully to {deployment_target} with ID {deployment_id}")
            return {
                "status": "success",
                "deployment_id": deployment_id,
                "target": deployment_target,
                "model_path": model_path,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to deploy model: {str(e)}")
            return {"status": "failed", "reason": str(e), "metrics": metrics}


class MonitoringService:
    """Handles model monitoring after deployment."""
    
    def __init__(self, config: MLOpsConfig):
        """Initialize with configuration.
        
        Args:
            config: MLOps configuration instance
        """
        self.config = config
        self.monitoring_config = config.get("monitoring", {})
        
    def setup(self, deployment_info: Dict[str, Any]) -> None:
        """Set up monitoring for a deployed model.
        
        Args:
            deployment_info: Dictionary with deployment information
        """
        if deployment_info.get("status") != "success":
            logger.warning("Cannot set up monitoring for unsuccessful deployment")
            return
            
        monitoring_metrics = self.monitoring_config.get("metrics", [])
        alert_thresholds = self.monitoring_config.get("alert_thresholds", {})
        
        logger.info(f"Setting up monitoring for deployment {deployment_info.get('deployment_id')}")
        logger.info(f"Monitoring metrics: {monitoring_metrics}")
        logger.info(f"Alert thresholds: {alert_thresholds}")
        
        # In a real implementation, this would:
        # 1. Configure monitoring tools
        # 2. Set up data collection
        # 3. Configure alerting
    
    def check_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """Check current metrics for a deployed model.
        
        In a real system, this would pull metrics from a monitoring system.
        This mock implementation simulates metric drift.
        
        Args:
            deployment_id: ID of the deployment to check
            
        Returns:
            Dictionary with monitoring results
        """
        # For demonstration, simulate metric drift
        # In a real system, this would query actual monitoring data
        
        alert_thresholds = self.monitoring_config.get("alert_thresholds", {})
        
        # Simulate metric drift
        current_metrics = {
            "accuracy": 0.78,  # Degraded from initial ~0.83
            "latency_ms": 120,
            "data_drift_score": 0.15
        }
        
        alerts = []
        
        for metric_name, threshold in alert_thresholds.items():
            if metric_name in current_metrics:
                # Different metrics may have different comparison operators
                if metric_name == "latency_ms" and current_metrics[metric_name] > threshold:
                    alerts.append(f"{metric_name} above threshold: {current_metrics[metric_name]} > {threshold}")
                elif metric_name != "latency_ms" and current_metrics[metric_name] < threshold:
                    alerts.append(f"{metric_name} below threshold: {current_metrics[metric_name]} < {threshold}")
        
        needs_retraining = len(alerts) > 0
        logger.info(f"Monitoring check for {deployment_id}: {len(alerts)} alerts, needs_retraining={needs_retraining}")
        
        return {
            "deployment_id": deployment_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": current_metrics,
            "alerts": alerts,
            "needs_retraining": needs_retraining
        }


class ExperimentTracker:
    """Tracks experiments and metrics."""
    
    def __init__(self, config: MLOpsConfig):
        """Initialize with configuration.
        
        Args:
            config: MLOps configuration instance
        """
        self.config = config
        self.tracking_config = config.get("experiment_tracking", {})
        self.current_run_id = None
        self.metrics = {}
        self.parameters = {}
        
    def start_run(self) -> str:
        """Start a new experiment run.
        
        Returns:
            Run ID
        """
        self.current_run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics = {}
        self.parameters = {}
        logger.info(f"Started experiment run {self.current_run_id}")
        return self.current_run_id
        
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics for the current run.
        
        Args:
            metrics: Dictionary with metrics
        """
        if not self.current_run_id:
            logger.warning("Cannot log metrics without an active run")
            return
            
        self.metrics.update(metrics)
        logger.info(f"Logging metrics for run {self.current_run_id}: {metrics}")
        
        # In a real implementation, this would save metrics to a tracking system
    
    def log_parameter(self, name: str, value: Any) -> None:
        """Log a parameter for the current run.
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        if not self.current_run_id:
            logger.warning("Cannot log parameter without an active run")
            return
            
        self.parameters[name] = value
        logger.info(f"Logging parameter for run {self.current_run_id}: {name}={value}")
        
    def log_failure(self, error_message: str) -> None:
        """Log a failure for the current run.
        
        Args:
            error_message: Error message
        """
        if not self.current_run_id:
            logger.warning("Cannot log failure without an active run")
            return
            
        logger.info(f"Logging failure for run {self.current_run_id}: {error_message}")
        
    def end_run(self) -> None:
        """End the current run."""
        if not self.current_run_id:
            logger.warning("No active run to end")
            return
            
        logger.info(f"Ended experiment run {self.current_run_id}")
        self.current_run_id = None


class MLOpsPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: str):
        """Initialize the pipeline.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = MLOpsConfig(config_path)
        
        # Initialize pipeline stages
        self.data_ingestion = DataIngestionStage(self.config)
        self.data_preprocessing = DataPreprocessingStage(self.config)
        self.model_training = ModelTrainingStage(self.config)
        self.model_evaluation = ModelEvaluationStage(self.config)
        self.model_deployment = ModelDeploymentStage(self.config)
        self.monitoring = MonitoringService(self.config)
        
        # Retraining configuration
        self.retraining_config = self.config.get("retraining", {})
        self.auto_retrain = self.retraining_config.get("auto_retrain", True)
        self.max_retrain_attempts = self.retraining_config.get("max_attempts", 3)
        
    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        try:
            logger.info("Starting MLOps pipeline")
            
            # Execute pipeline stages
            ingested_data = self.data_ingestion.execute()
            processed_data = self.data_preprocessing.execute(ingested_data)
            
            # Split data for training and testing (simplified)
            train_data = {k: v for k, v in processed_data.items()}
            test_data = {k: v for k, v in processed_data.items()}
            
            # First training attempt
            model_path = self.model_training.execute(train_data)
            metrics = self.model_evaluation.execute(model_path, test_data)
            
            # Check if retraining is needed and auto-retrain is enabled
            retrain_attempts = 0
            retraining_history = []
            
            while (not metrics.get("all_metrics_pass", False) and 
                  self.auto_retrain and 
                  retrain_attempts < self.max_retrain_attempts):
                
                retrain_attempts += 1
                logger.info(f"Model metrics below threshold, starting retraining (attempt {retrain_attempts}/{self.max_retrain_attempts})")
                
                # Log failing metrics
                failing_metrics = metrics.get("failing_metrics", {})
                for metric_name, details in failing_metrics.items():
                    logger.info(f"Metric requiring improvement: {metric_name} = {details['value']} (threshold: {details['threshold']})")
                
                # Store previous attempt info
                retraining_history.append({
                    "attempt": retrain_attempts,
                    "model_path": model_path,
                    "metrics": metrics
                })
                
                # Execute retraining
                model_path = self.model_training.execute(train_data, is_retraining=True)
                metrics = self.model_evaluation.execute(model_path, test_data)
                
                logger.info(f"Retraining attempt {retrain_attempts} complete. All metrics pass: {metrics.get('all_metrics_pass', False)}")
            
            # Attempt to deploy the model (original or retrained)
            deployment_info = self.model_deployment.execute(model_path, metrics)
            
            if deployment_info.get("status") == "success":
                self.monitoring.setup(deployment_info)
                
                # Simulate monitoring check that triggers retraining
                if self.config.get("simulate_monitoring_check", False):
                    logger.info("Simulating monitoring check after deployment")
                    monitoring_result = self.monitoring.check_metrics(deployment_info.get("deployment_id"))
                    
                    if monitoring_result.get("needs_retraining"):
                        logger.info("Monitoring detected metrics below threshold, retraining recommended")
                        # In a real system, this might trigger an automatic retraining job
            
            logger.info("MLOps pipeline completed successfully")
            
            return {
                "ingested_data": ingested_data,
                "processed_data": processed_data,
                "model_path": model_path,
                "metrics": metrics,
                "retraining_history": retraining_history,
                "retraining_attempts": retrain_attempts,
                "deployment": deployment_info
            }
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def run_model_monitoring(self, deployment_id: str) -> Dict[str, Any]:
        """Run model monitoring for a deployed model.
        
        Args:
            deployment_id: ID of the deployment to monitor
            
        Returns:
            Dictionary with monitoring results and actions taken
        """
        logger.info(f"Running monitoring check for deployment {deployment_id}")
        
        try:
            # Check current metrics
            monitoring_result = self.monitoring.check_metrics(deployment_id)
            
            # If retraining is needed, trigger retraining pipeline
            if monitoring_result.get("needs_retraining") and self.auto_retrain:
                logger.info("Metrics below threshold detected, triggering retraining pipeline")
                
                # In a real system, this would trigger the retraining pipeline
                # For this example, we'll just log the intent
                return {
                    "monitoring_result": monitoring_result,
                    "action": "retraining_triggered"
                }
            
            return {
                "monitoring_result": monitoring_result,
                "action": "none_required"
            }
        except Exception as e:
            logger.error(f"Monitoring check failed: {str(e)}")
            return {
                "status": "failed",
                "reason": str(e)
            }


def main():
    """Entry point for the MLOps framework."""
    parser = argparse.ArgumentParser(description="MLOps Framework")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--monitor", help="Run monitoring check for the specified deployment ID")
    args = parser.parse_args()
    
    try:
        pipeline = MLOpsPipeline(args.config)
        
        if args.monitor:
            # Run monitoring check only
            results = pipeline.run_model_monitoring(args.monitor)
            print(f"Monitoring results: {results}")
        else:
            # Run full pipeline
            results = pipeline.run()
            print(f"Pipeline results: {results}")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
