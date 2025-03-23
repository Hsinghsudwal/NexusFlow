# pipeline/data_pipeline.py - Data processing pipeline
from typing import Dict, Any, Optional, List
import pandas as pd
import os
import datetime

from ..pipeline.base import Pipeline
from ..components.base import Component
from ..config import MLOpsConfig

class DataPipeline(Pipeline):
    """Pipeline for data processing."""
    
    def __init__(self, name: str, config: MLOpsConfig):
        super().__init__(name, config)
        self.metadata["pipeline_type"] = "data"
    
    def run(self, data_source: Any, **kwargs) -> Dict[str, Any]:
        """Run the data pipeline."""
        self.logger.info("Starting data pipeline")
        self.metadata["status"] = "running"
        self.metadata["started_at"] = datetime.datetime.now().isoformat()
        
        current_data = data_source
        
        # Execute each component
        for component in self.components:
            self.logger.info(f"Executing component: {component.name}")
            current_data = component.execute(current_data)
        
        # Save final data artifact
        self.save_artifact("processed_data", current_data, "data")
        
        self.metadata["status"] = "completed"
        self.metadata["completed_at"] = datetime.datetime.now().isoformat()
        self.save_metadata()
        
        self.logger.info("Data pipeline completed")
        
        return {
            "data": current_data,
            "metadata": self.metadata,
            "artifacts": self.artifacts
        }

# pipeline/training_pipeline.py - Model training pipeline
from typing import Dict, Any, Optional, List
import pandas as pd

class TrainingPipeline(Pipeline):
    """Pipeline for model training."""
    
    def __init__(self, name: str, config: MLOpsConfig):
        super().__init__(name, config)
        self.metadata["pipeline_type"] = "training"
    
    def run(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Run the training pipeline."""
        self.logger.info("Starting training pipeline")
        self.metadata["status"] = "running"
        self.metadata["started_at"] = datetime.datetime.now().isoformat()
        
        # Process through each component
        features = None
        model = None
        
        for component in self.components:
            self.logger.info(f"Executing component: {component.name}")
            
            if hasattr(component, "component_type"):
                if component.component_type == "feature_engineering":
                    features = component.execute(data)
                    self.save_artifact("features", features, "feature")
                elif component.component_type == "model_training":
                    model = component.execute(features)
                    self.save_artifact("model", model, "model")
            else:
                # Generic component
                result = component.execute(data)
                
                if result is not None:
                    if hasattr(result, "predict"):
                        model = result
                        self.save_artifact("model", model, "model")
        
        self.metadata["status"] = "completed"
        self.metadata["completed_at"] = datetime.datetime.now().isoformat()
        self.save_metadata()
        
        self.logger.info("Training pipeline completed")
        
        return {
            "model": model,
            "features": features,
            "metadata": self.metadata,
            "artifacts": self.artifacts
        }

# pipeline/evaluation_pipeline.py - Model evaluation pipeline
from typing import Dict, Any, Optional, List
import pandas as pd
import json

class EvaluationPipeline(Pipeline):
    """Pipeline for model evaluation."""
    
    def __init__(self, name: str, config: MLOpsConfig):
        super().__init__(name, config)
        self.metadata["pipeline_type"] = "evaluation"
    
    def run(self, model: Any, test_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Run the evaluation pipeline."""
        self.logger.info("Starting evaluation pipeline")
        self.metadata["status"] = "running"
        self.metadata["started_at"] = datetime.datetime.now().isoformat()
        
        metrics = {}
        
        # Execute each component
        for component in self.components:
            self.logger.info(f"Executing component: {component.name}")
            component_metrics = component.execute(model, test_data)
            
            if component_metrics:
                metrics.update(component_metrics)
        
        # Save metrics artifact
        self.save_artifact("metrics", metrics, "metrics")
        
        # Update pipeline metadata
        self.metadata["status"] = "completed"
        self.metadata["completed_at"] = datetime.datetime.now().isoformat()
        self.metadata["metrics"] = metrics
        self.save_metadata()
        
        self.logger.info("Evaluation pipeline completed")
        
        return {
            "metrics": metrics,
            "metadata": self.metadata,
            "artifacts": self.artifacts
        }

# pipeline/deployment_pipeline.py - Model deployment pipeline
from typing import Dict, Any, Optional
import os
import json
import datetime

class DeploymentPipeline(Pipeline):
    """Pipeline for model deployment."""
    
    def __init__(self, name: str, config: MLOpsConfig):
        super().__init__(name, config)
        self.metadata["pipeline_type"] = "deployment"
    
    def run(self, model: Any, metrics: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run the deployment pipeline."""
        self.logger.info("Starting deployment pipeline")
        self.metadata["status"] = "running"
        self.metadata["started_at"] = datetime.datetime.now().isoformat()
        
        # Register model in registry
        from ..versioning.model_registry import ModelRegistry
        registry = ModelRegistry(self.config)
        
        model_name = kwargs.get("model_name", "default_model")
        model_version = kwargs.get("model_version", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        
        # First save model as artifact
        model_path = self.save_artifact("deployment_model", model, "model")
        
        # Register model with metadata
        model_metadata = {
            "model_name": model_name,
            "model_version": model_version,
            "metrics": metrics,
            "pipeline_id": self.id,
            "created_at": datetime.datetime.now().isoformat(),
            "created_by": kwargs.get("created_by", "unknown")
        }
        
        registry_path = registry.register_model(
            model_path=model_path,
            model_name=model_name,
            version=model_version,
            metadata=model_metadata
        )
        
        # Execute deployment components
        deployment_info = {}
        
        for component in self.components:
            self.logger.info(f"Executing component: {component.name}")
            result = component.execute(
                model=model,
                model_path=registry_path,
                model_name=model_name,
                model_version=model_version,
                **kwargs
            )
            
            if result:
                deployment_info.update(result)
        
        # Save deployment info
        self.save_artifact("deployment_info", deployment_info, "config")
        
        # Promote to production if specified
        if kwargs.get("promote_to_production", False):
            registry.promote_model_to_production(model_name, model_version)
            self.metadata["promoted_to_production"] = True
        
        self.metadata["status"] = "completed"
        self.metadata["completed_at"] = datetime.datetime.now().isoformat()
        self.metadata["deployment_info"] = deployment_info
        self.save_metadata()
        
        self.logger.info("Deployment pipeline completed")
        
        return {
            "deployment_info": deployment_info,
            "metadata": self.metadata,
            "artifacts": self.artifacts
        }

# monitoring/data_drift.py - Data drift detection
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import json
import os

class DataDriftMonitor:
    """Monitor data drift in production data."""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.logger = setup_logger("data_drift_monitor")
    
    def calculate_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: List[str],
        drift_method: str = "ks_test"
    ) -> Dict[str, Any]:
        """Calculate drift between reference and current data."""
        self.logger.info(f"Calculating data drift using {drift_method}")
        
        drift_metrics = {}
        
        if drift_method == "ks_test":
            from scipy import stats
            
            for feature in features:
                if feature in reference_data.columns and feature in current_data.columns:
                    # Skip non-numeric columns for KS test
                    if not np.issubdtype(reference_data[feature].dtype, np.number):
                        continue
                    
                    # Calculate KS test
                    ks_stat, p_value = stats.ks_2samp(
                        reference_data[feature].dropna(),
                        current_data[feature].dropna()
                    )
                    
                    drift_metrics[feature] = {
                        "drift_score": float(ks_stat),
                        "p_value": float(p_value),
                        "drift_detected": p_value < 0.05
                    }
        
        elif drift_method == "distribution_difference":
            for feature in features:
                if feature in reference_data.columns and feature in current_data.columns:
                    # Calculate simple distribution difference
                    ref_mean = float(reference_data[feature].mean())
                    curr_mean = float(current_data[feature].mean())
                    
                    # Mean shift as percentage
                    if ref_mean != 0:
                        mean_shift_pct = abs(ref_mean - curr_mean) / abs(ref_mean) * 100
                    else:
                        mean_shift_pct = float('inf') if curr_mean != 0 else 0
                    
                    drift_metrics[feature] = {
                        "ref_mean": ref_mean,
                        "current_mean": curr_mean,
                        "mean_shift_pct": float(mean_shift_pct),
                        "drift_detected": mean_shift_pct > 15  # Threshold for drift
                    }
        
        summary = {
            "drift_method": drift_method,
            "total_features": len(features),
            "features_with_drift": sum(1 for f in drift_metrics if drift_metrics[f].get("drift_detected", False)),
            "drift_timestamp": datetime.datetime.now().isoformat()
        }
        
        return {
            "drift_metrics": drift_metrics,
            "summary": summary
        }
    
    def save_drift_report(
        self,
        drift_results: Dict[str, Any],
        model_name: str,
        model_version: str
    ) -> str:
        """Save drift report to disk."""
        # Create report path
        report_dir = os.path.join(
            self.config.artifact_store_path,
            "monitoring",
            "data_drift",
            model_name,
            model_version
        )
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        report_path = os.path.join(report_dir, f"drift_report_{timestamp}.json")
        
        with open(report_path, 'w') as f:
            json.dump(drift_results, f, indent=2)
        
        self.logger.info(f"Saved drift report to {report_path}")
        return report_path

# monitoring/model_drift.py - Model performance drift
class ModelDriftMonitor:
    """Monitor model performance drift in production."""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.logger = setup_logger("model_drift_monitor")
    
    def calculate_performance_drift(
        self,
        reference_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        threshold_pct: float = 10.0
    ) -> Dict[str, Any]:
        """Calculate performance drift between reference and current metrics."""
        self.logger.info("Calculating model performance drift")
        
        drift_metrics = {}
        
        for metric_name, ref_value in reference_metrics.items():
            if metric_name in current_metrics:
                curr_value = current_metrics[metric_name]
                
                # Calculate drift as percentage
                if ref_value != 0:
                    drift_pct = abs(ref_value - curr_value) / abs(ref_value) * 100
                else:
                    drift_pct = float('inf') if curr_value != 0 else 0
                
                drift_metrics[metric_name] = {
                    "reference_value": float(ref_value),
                    "current_value": float(curr_value),
                    "drift_percentage": float(drift_pct),
                    "drift_detected": drift_pct > threshold_pct
                }
        
        summary = {
            "total_metrics": len(reference_metrics),
            "metrics_with_drift": sum(1 for m in drift_metrics if drift_metrics[m]["drift_detected"]),
            "threshold_percentage": threshold_pct,
            "drift_timestamp": datetime.datetime.now().isoformat()
        }
        
        return {
            "drift_metrics": drift_metrics,
            "summary": summary
        }
    
    def save_performance_drift_report(
        self,
        drift_results: Dict[str, Any],
        model_name: str,
        model_version: str
    ) -> str:
        """Save performance drift report to disk."""
        # Create report path
        report_dir = os.path.join(
            self.config.artifact_store_path,
            "monitoring",
            "model_drift",
            model_name,
            model_version
        )
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        report_path = os.path.join(report_dir, f"performance_drift_report_{timestamp}.json")
        
        with open(report_path, 'w') as f:
            json.dump(drift_results, f, indent=2)
        
        self.logger.info(f"Saved performance drift report to {report_path}")
        return report_path

# monitoring/alerting.py - Alerting system
from enum import Enum
import smtplib
from email