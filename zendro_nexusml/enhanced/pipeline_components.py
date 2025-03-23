# Core pipeline component class
class PipelineComponent:
    def __init__(self, name, version="0.1.0"):
        self.name = name
        self.version = version
        self.metadata = {}
        
    def execute(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement execute method")
        
    def log_metadata(self, key, value):
        """Store metadata about component execution"""
        self.metadata[key] = value
        
# Example pipeline component implementation
class DataPreprocessor(PipelineComponent):
    def __init__(self, name, transformations=None, version="0.1.0"):
        super().__init__(name, version)
        self.transformations = transformations or []
        
    def execute(self, data):
        # Log input shape as metadata
        self.log_metadata("input_shape", data.shape)
        
        # Apply each transformation
        for transform in self.transformations:
            data = transform(data)
            
        # Log output shape
        self.log_metadata("output_shape", data.shape)
        return data

# ======managemnet
import uuid
import datetime
import json

class Pipeline:
    def __init__(self, name, components=None, version="0.1.0"):
        self.name = name
        self.version = version
        self.components = components or []
        self.id = str(uuid.uuid4())
        self.metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat()
        }
        
    def add_component(self, component):
        """Add a component to the pipeline"""
        self.components.append(component)
        self.metadata["last_updated"] = datetime.datetime.now().isoformat()
        
    def execute(self, input_data):
        """Execute all components in sequence"""
        result = input_data
        execution_log = []
        
        for component in self.components:
            start_time = datetime.datetime.now()
            result = component.execute(result)
            end_time = datetime.datetime.now()
            
            # Log execution details
            execution_log.append({
                "component_name": component.name,
                "component_version": component.version,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_ms": (end_time - start_time).total_seconds() * 1000,
                "metadata": component.metadata
            })
            
        self.metadata["last_execution"] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "components": execution_log
        }
        
        return result
        
    def export_definition(self, filepath):
        """Export pipeline definition to JSON"""
        definition = {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "components": [
                {"name": c.name, "type": c.__class__.__name__, "version": c.version}
                for c in self.components
            ],
            "metadata": self.metadata
        }
        
        with open(filepath, "w") as f:
            json.dump(definition, f, indent=2)

# version and metadata registry
class MLRegistry:
    def __init__(self, storage_path="./ml_registry"):
        self.storage_path = storage_path
        self.pipelines = {}
        self.components = {}
        self.models = {}
        
    def register_pipeline(self, pipeline):
        """Register a pipeline in the registry"""
        pipeline_key = f"{pipeline.name}:{pipeline.version}"
        self.pipelines[pipeline_key] = {
            "id": pipeline.id,
            "definition": {
                "name": pipeline.name,
                "version": pipeline.version,
                "components": [
                    {"name": c.name, "type": c.__class__.__name__, "version": c.version}
                    for c in pipeline.components
                ],
            },
            "metadata": pipeline.metadata,
            "created_at": datetime.datetime.now().isoformat()
        }
        return pipeline_key
        
    def get_pipeline(self, name, version=None):
        """Retrieve pipeline by name and optional version"""
        if version:
            pipeline_key = f"{name}:{version}"
            return self.pipelines.get(pipeline_key)
        
        # If no version specified, get the latest
        matching_pipelines = [k for k in self.pipelines.keys() if k.startswith(f"{name}:")]
        if not matching_pipelines:
            return None
            
        # Sort by version and get the latest
        latest_key = sorted(matching_pipelines)[-1]
        return self.pipelines.get(latest_key)
        
    def save(self):
        """Save registry to disk"""
        os.makedirs(self.storage_path, exist_ok=True)
        
        with open(f"{self.storage_path}/registry.json", "w") as f:
            registry_data = {
                "pipelines": self.pipelines,
                "components": self.components,
                "models": self.models,
                "last_updated": datetime.datetime.now().isoformat()
            }
            json.dump(registry_data, f, indent=2)
            
    def load(self):
        """Load registry from disk"""
        registry_path = f"{self.storage_path}/registry.json"
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                registry_data = json.load(f)
                self.pipelines = registry_data.get("pipelines", {})
                self.components = registry_data.get("components", {})
                self.models = registry_data.get("models", {})






import os
import uuid
import json
import datetime
import logging
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
import pickle
import hashlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mlops-framework")

# -------------------- Core Components --------------------

class PipelineComponent(ABC):
    """Abstract base class for all pipeline components"""
    
    def __init__(self, name: str, version: str = "0.1.0", config: Dict = None):
        self.name = name
        self.version = version
        self.config = config or {}
        self.metadata = {}
        self.id = str(uuid.uuid4())
        self.input_schema = None
        self.output_schema = None
        
    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute the component's logic"""
        pass
        
    def log_metadata(self, key: str, value: Any):
        """Store metadata about component execution"""
        self.metadata[key] = value
        
    def validate_input(self, data: Any) -> bool:
        """Validate input data against schema if available"""
        if self.input_schema is None:
            return True
        try:
            self.input_schema.validate(data)
            return True
        except Exception as e:
            logger.error(f"Input validation failed for {self.name}: {str(e)}")
            return False
            
    def validate_output(self, data: Any) -> bool:
        """Validate output data against schema if available"""
        if self.output_schema is None:
            return True
        try:
            self.output_schema.validate(data)
            return True
        except Exception as e:
            logger.error(f"Output validation failed for {self.name}: {str(e)}")
            return False
            
    def to_dict(self) -> Dict:
        """Convert component to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "type": self.__class__.__name__,
            "config": self.config,
            "metadata": self.metadata
        }
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, version={self.version})"


class DataLoader(PipelineComponent):
    """Component for loading data from various sources"""
    
    def __init__(self, name: str, source_type: str, source_config: Dict, version: str = "0.1.0"):
        super().__init__(name, version, config={"source_type": source_type, "source_config": source_config})
        
    def execute(self, *args, **kwargs):
        """Load data from the configured source"""
        logger.info(f"Loading data using {self.name}")
        source_type = self.config.get("source_type")
        source_config = self.config.get("source_config", {})
        
        start_time = time.time()
        
        # Example implementation for different source types
        if source_type == "csv":
            import pandas as pd
            data = pd.read_csv(source_config.get("filepath"))
            
        elif source_type == "database":
            import pandas as pd
            import sqlalchemy
            engine = sqlalchemy.create_engine(source_config.get("connection_string"))
            data = pd.read_sql(source_config.get("query"), engine)
            
        elif source_type == "api":
            import requests
            import pandas as pd
            response = requests.get(
                source_config.get("url"),
                headers=source_config.get("headers", {}),
                params=source_config.get("params", {})
            )
            data = pd.DataFrame(response.json())
            
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
            
        # Log metadata
        self.log_metadata("rows_loaded", len(data))
        self.log_metadata("columns", list(data.columns))
        self.log_metadata("load_time_seconds", time.time() - start_time)
        
        return data


class DataPreprocessor(PipelineComponent):
    """Component for data preprocessing and feature engineering"""
    
    def __init__(self, name: str, transformations: List[Callable] = None, version: str = "0.1.0", 
                 config: Dict = None):
        super().__init__(name, version, config=config or {})
        self.transformations = transformations or []
        
    def add_transformation(self, transform_func: Callable):
        """Add a transformation function to the processor"""
        self.transformations.append(transform_func)
        
    def execute(self, data: Any, *args, **kwargs):
        """Apply all transformations to the input data"""
        logger.info(f"Preprocessing data using {self.name}")
        
        # Validate input
        if not self.validate_input(data):
            raise ValueError(f"Input validation failed for {self.name}")
            
        # Log input metadata
        if hasattr(data, "shape"):
            self.log_metadata("input_shape", data.shape)
            
        # Apply each transformation
        start_time = time.time()
        transformation_logs = []
        
        for i, transform in enumerate(self.transformations):
            transform_start = time.time()
            data = transform(data)
            transform_end = time.time()
            
            transformation_logs.append({
                "step": i,
                "name": transform.__name__ if hasattr(transform, "__name__") else f"transform_{i}",
                "duration_seconds": transform_end - transform_start
            })
            
        # Log output metadata
        if hasattr(data, "shape"):
            self.log_metadata("output_shape", data.shape)
            
        self.log_metadata("total_processing_time", time.time() - start_time)
        self.log_metadata("transformations", transformation_logs)
        
        # Validate output
        if not self.validate_output(data):
            raise ValueError(f"Output validation failed for {self.name}")
            
        return data


class ModelTrainer(PipelineComponent):
    """Component for training machine learning models"""
    
    def __init__(self, name: str, model_type: str, hyperparameters: Dict = None, 
                 version: str = "0.1.0", config: Dict = None):
        super().__init__(name, version, config=config or {})
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self.model = None
        
    def execute(self, data: Any, target_column: str = None, features: List[str] = None, 
                *args, **kwargs):
        """Train a model on the provided data"""
        logger.info(f"Training model using {self.name}")
        
        if target_column is None and "target_column" in self.config:
            target_column = self.config["target_column"]
            
        if features is None and "features" in self.config:
            features = self.config["features"]
            
        if target_column is None:
            raise ValueError("Target column must be specified")
            
        # Prepare features and target
        if features:
            X = data[features]
        else:
            X = data.drop(columns=[target_column])
            features = X.columns.tolist()
            
        y = data[target_column]
        
        # Log input data stats
        self.log_metadata("training_samples", len(X))
        self.log_metadata("features", features)
        self.log_metadata("target_column", target_column)
        
        # Initialize and train model
        start_time = time.time()
        
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if self.config.get("task") == "regression":
                self.model = RandomForestRegressor(**self.hyperparameters)
            else:
                self.model = RandomForestClassifier(**self.hyperparameters)
                
        elif self.model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            if self.config.get("task") == "regression":
                self.model = GradientBoostingRegressor(**self.hyperparameters)
            else:
                self.model = GradientBoostingClassifier(**self.hyperparameters)
                
        elif self.model_type == "linear":
            from sklearn.linear_model import LinearRegression, LogisticRegression
            if self.config.get("task") == "regression":
                self.model = LinearRegression(**self.hyperparameters)
            else:
                self.model = LogisticRegression(**self.hyperparameters)
                
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        # Train the model
        self.model.fit(X, y)
        
        # Log training metadata
        training_time = time.time() - start_time
        self.log_metadata("training_time_seconds", training_time)
        
        # Log model parameters if possible
        if hasattr(self.model, "get_params"):
            self.log_metadata("model_parameters", self.model.get_params())
            
        # Log feature importance if available
        if hasattr(self.model, "feature_importances_"):
            feature_importance = dict(zip(features, self.model.feature_importances_))
            self.log_metadata("feature_importance", feature_importance)
            
        # Create model artifact
        model_artifact = {
            "model": self.model,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "features": features,
            "target_column": target_column,
            "metadata": self.metadata,
            "trained_at": datetime.datetime.now().isoformat()
        }
        
        return model_artifact


class ModelEvaluator(PipelineComponent):
    """Component for evaluating model performance"""
    
    def __init__(self, name: str, metrics: List[str] = None, version: str = "0.1.0", 
                 config: Dict = None):
        super().__init__(name, version, config=config or {})
        self.metrics = metrics or ["accuracy", "f1"]
        
    def execute(self, model_artifact: Dict, test_data: Any, 
                target_column: str = None, *args, **kwargs):
        """Evaluate model performance on test data"""
        logger.info(f"Evaluating model using {self.name}")
        
        model = model_artifact["model"]
        features = model_artifact["features"]
        
        if target_column is None:
            target_column = model_artifact["target_column"]
            
        # Prepare test data
        X_test = test_data[features]
        y_test = test_data[target_column]
        
        # Make predictions
        task = self.config.get("task", "classification")
        if task == "classification" and hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
            self.log_metadata("has_probability", True)
        else:
            self.log_metadata("has_probability", False)
            
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics_results = {}
        
        if task == "classification":
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.metrics import roc_auc_score, confusion_matrix
            
            if "accuracy" in self.metrics:
                metrics_results["accuracy"] = float(accuracy_score(y_test, y_pred))
                
            if "precision" in self.metrics:
                metrics_results["precision"] = float(precision_score(y_test, y_pred, average="weighted"))
                
            if "recall" in self.metrics:
                metrics_results["recall"] = float(recall_score(y_test, y_pred, average="weighted"))
                
            if "f1" in self.metrics:
                metrics_results["f1"] = float(f1_score(y_test, y_pred, average="weighted"))
                
            if "roc_auc" in self.metrics and self.log_metadata.get("has_probability", False):
                # For multi-class, calculate ROC AUC for each class
                if len(set(y_test)) > 2:
                    metrics_results["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba, multi_class="ovr"))
                else:
                    metrics_results["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
                    
            if "confusion_matrix" in self.metrics:
                cm = confusion_matrix(y_test, y_pred)
                metrics_results["confusion_matrix"] = cm.tolist()
                
        elif task == "regression":
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            if "mse" in self.metrics:
                metrics_results["mse"] = float(mean_squared_error(y_test, y_pred))
                
            if "rmse" in self.metrics:
                metrics_results["rmse"] = float(mean_squared_error(y_test, y_pred, squared=False))
                
            if "mae" in self.metrics:
                metrics_results["mae"] = float(mean_absolute_error(y_test, y_pred))
                
            if "r2" in self.metrics:
                metrics_results["r2"] = float(r2_score(y_test, y_pred))
                
        # Log all metrics
        self.log_metadata("metrics", metrics_results)
        self.log_metadata("test_samples", len(X_test))
        
        # Add evaluation results to model artifact
        model_artifact["evaluation"] = {
            "metrics": metrics_results,
            "test_samples": len(X_test),
            "evaluated_at": datetime.datetime.now().isoformat()
        }
        
        return model_artifact


class ModelSerializer(PipelineComponent):
    """Component for serializing and saving trained models"""
    
    def __init__(self, name: str, output_dir: str = "./models", version: str = "0.1.0", 
                 format: str = "pickle", config: Dict = None):
        super().__init__(name, version, config=config or {})
        self.output_dir = output_dir
        self.format = format
        
    def execute(self, model_artifact: Dict, *args, **kwargs):
        """Serialize and save the model"""
        logger.info(f"Serializing model using {self.name}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a unique model filename
        model_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_artifact.get("model_type", "model")
        filename = f"{model_name}_{timestamp}_{model_id}"
        
        # Save model metadata separate from the model binary
        metadata = {
            "id": model_id,
            "name": model_name,
            "type": model_artifact["model_type"],
            "hyperparameters": model_artifact["hyperparameters"],
            "features": model_artifact["features"],
            "target_column": model_artifact["target_column"],
            "trained_at": model_artifact.get("trained_at"),
            "evaluation": model_artifact.get("evaluation"),
            "metadata": model_artifact.get("metadata", {}),
            "serialized_at": datetime.datetime.now().isoformat(),
            "format": self.format
        }
        
        # Save model based on format
        if self.format == "pickle":
            model_path = os.path.join(self.output_dir, f"{filename}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model_artifact["model"], f)
                
        elif self.format == "joblib":
            import joblib
            model_path = os.path.join(self.output_dir, f"{filename}.joblib")
            joblib.dump(model_artifact["model"], model_path)
            
        elif self.format == "onnx":
            # Would require specific conversion based on model type
            # This is a placeholder for ONNX conversion
            try:
                import onnxmltools
                # Convert based on model type - simplified example
                model_path = os.path.join(self.output_dir, f"{filename}.onnx")
                onnx_model = onnxmltools.convert_sklearn(model_artifact["model"])
                onnxmltools.utils.save_model(onnx_model, model_path)
            except ImportError:
                raise ImportError("onnxmltools is required for ONNX format")
                
        else:
            raise ValueError(f"Unsupported serialization format: {self.format}")
            
        # Save metadata
        metadata_path = os.path.join(self.output_dir, f"{filename}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Log paths
        self.log_metadata("model_path", model_path)
        self.log_metadata("metadata_path", metadata_path)
        self.log_metadata("model_id", model_id)
        
        # Return updated artifact with paths
        model_artifact["model_path"] = model_path
        model_artifact["metadata_path"] = metadata_path
        model_artifact["model_id"] = model_id
        
        return model_artifact


# -------------------- Pipeline Framework --------------------

class Pipeline:
    """Core pipeline class for managing a sequence of components"""
    
    def __init__(self, name: str, components: List[PipelineComponent] = None, 
                 version: str = "0.1.0", description: str = ""):
        self.name = name
        self.version = version
        self.description = description
        self.components = components or []
        self.id = str(uuid.uuid4())
        self.metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat()
        }
        self.results = {}
        
    def add_component(self, component: PipelineComponent):
        """Add a component to the pipeline"""
        self.components.append(component)
        self.metadata["last_updated"] = datetime.datetime.now().isoformat()
        return self
        
    def execute(self, input_data: Any = None, ctx: Dict = None):
        """Execute all components in sequence"""
        logger.info(f"Executing pipeline {self.name} v{self.version}")
        
        if ctx is None:
            ctx = {}
            
        result = input_data
        execution_log = []
        pipeline_start_time = time.time()
        
        # Execute each component in sequence
        for idx, component in enumerate(self.components):
            logger.info(f"Running component {idx+1}/{len(self.components)}: {component.name}")
            
            start_time = time.time()
            
            try:
                # Pass both previous result and context data
                component_args = ctx.get(f"component_{idx}_args", [])
                component_kwargs = ctx.get(f"component_{idx}_kwargs", {})
                
                if result is not None and "data" not in component_kwargs:
                    # Default behavior: pass previous result as first argument
                    if len(component_args) > 0:
                        component_args = list(component_args)
                        component_args[0] = result
                    else:
                        component_args = [result]
                
                # Execute the component
                result = component.execute(*component_args, **component_kwargs)
                
                # Store intermediate result if requested
                if ctx.get("store_intermediate_results", False):
                    self.results[component.name] = result
                    
                end_time = time.time()
                status = "success"
                error = None
                
            except Exception as e:
                end_time = time.time()
                status = "error"
                error = str(e)
                logger.error(f"Error in component {component.name}: {str(e)}")
                
                if not ctx.get("continue_on_error", False):
                    raise
                    
            # Log execution details
            execution_log.append({
                "component_id": component.id,
                "component_name": component.name,
                "component_type": component.__class__.__name__,
                "component_version": component.version,
                "start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.datetime.fromtimestamp(end_time).isoformat(),
                "duration_seconds": end_time - start_time,
                "status": status,
                "error": error,
                "metadata": component.metadata.copy()
            })
            
        # Record overall execution metadata
        self.metadata["last_execution"] = {
            "start_time": datetime.datetime.fromtimestamp(pipeline_start_time).isoformat(),
            "end_time": datetime.datetime.fromtimestamp(time.time()).isoformat(),
            "total_duration_seconds": time.time() - pipeline_start_time,
            "components": execution_log
        }
        
        return result
        
    def export_definition(self, filepath: str = None):
        """Export pipeline definition to JSON"""
        definition = {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "components": [c.to_dict() for c in self.components],
            "metadata": self.metadata
        }
        
        if filepath:
            with open(filepath, "w") as f:
                json.dump(definition, f, indent=2)
                
        return definition
        
    def get_component_by_name(self, name: str) -> Optional[PipelineComponent]:
        """Get a component by name"""
        for component in self.components:
            if component.name == name:
                return component
        return None
        
    def visualize(self, filepath: str = None):
        """Create a simple visualization of the pipeline (text-based)"""
        viz = f"Pipeline: {self.name} (v{self.version})\n"
        viz += "=" * 50 + "\n"
        
        for i, component in enumerate(self.components):
            viz += f"{i+1}. [{component.__class__.__name__}] {component.name} (v{component.version})\n"
            if i < len(self.components) - 1:
                viz += "   |\n   v\n"
                
        viz += "=" * 50
        
        if filepath:
            with open(filepath, "w") as f:
                f.write(viz)
                
        return viz


# -------------------- Registry and Versioning --------------------

class MLRegistry:
    """Registry for tracking ML pipelines, components, and models"""
    
    def __init__(self, storage_path: str = "./ml_registry"):
        self.storage_path = storage_path
        self.pipelines = {}
        self.components = {}
        self.models = {}
        self.experiments = {}
        
        # Create storage directories if they don't exist
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "pipelines"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "experiments"), exist_ok=True)
        
    def register_pipeline(self, pipeline: Pipeline) -> str:
        """Register a pipeline in the registry"""
        pipeline_key = f"{pipeline.name}:{pipeline.version}"
        
        # Store pipeline definition
        self.pipelines[pipeline_key] = {
            "id": pipeline.id,
            "definition": pipeline.export_definition(),
            "registered_at": datetime.datetime.now().isoformat()
        }
        
        # Save to disk
        self.save_pipeline(pipeline_key)
        
        logger.info(f"Registered pipeline: {pipeline_key}")
        return pipeline_key
        
    def register_model(self, model_artifact: Dict) -> str:
        """Register a model in the registry"""
        model_id = model_artifact.get("model_id", str(uuid.uuid4()))
        
        # Extract key metadata
        model_info = {
            "id": model_id,
            "name": model_artifact.get("name", "unnamed_model"),
            "type": model_artifact.get("model_type"),
            "version": model_artifact.get("version", "0.1.0"),
            "path": model_artifact.get("model_path"),
            "metadata_path": model_artifact.get("metadata_path"),
            "created_at": model_artifact.get("trained_at", datetime.datetime.now().isoformat()),
            "registered_at": datetime.datetime.now().isoformat(),
            "metrics": model_artifact.get("evaluation", {}).get("metrics", {}),
            "features": model_artifact.get("features", []),
            "pipeline_id": model_artifact.get("pipeline_id")
        }
        
        #