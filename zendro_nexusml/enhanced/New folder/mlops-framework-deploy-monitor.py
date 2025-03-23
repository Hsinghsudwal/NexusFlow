# mlops_framework/
# ├── __init__.py
# ├── core/
# │   ├── __init__.py
# │   ├── pipeline.py
# │   ├── step.py
# │   └── config.py
# ├── components/
# │   ├── __init__.py
# │   ├── data_ingestion.py
# │   ├── data_validation.py
# │   ├── preprocessing.py
# │   ├── training.py
# │   ├── evaluation.py
# │   └── serving.py
# ├── storage/
# │   ├── __init__.py
# │   ├── local.py
# │   ├── s3.py
# │   └── artifact.py
# ├── orchestration/
# │   ├── __init__.py
# │   ├── local.py
# │   ├── airflow.py
# │   └── kubeflow.py
# ├── deployment/
# │   ├── __init__.py
# │   ├── docker.py
# │   ├── kubernetes.py
# │   └── serverless.py
# └── monitoring/
#     ├── __init__.py
#     ├── metrics.py
#     ├── logging.py
#     └── alerting.py

# core/pipeline.py
import os
import yaml
import uuid
import datetime
from typing import List, Dict, Any, Optional
from .step import Step
from ..storage.artifact import Artifact


class Pipeline:
    """Defines an ML pipeline that chains together multiple steps."""
    
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description
        self.steps = []
        self.uuid = str(uuid.uuid4())
        self.metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat()
        }
        
    def add_step(self, step: Step) -> None:
        """Add a step to the pipeline."""
        if not isinstance(step, Step):
            raise TypeError("Step must be an instance of Step class")
        self.steps.append(step)
        
    def run(self, config: Dict[str, Any]) -> Dict[str, Artifact]:
        """Run the entire pipeline and return artifacts."""
        artifacts = {}
        step_outputs = {}
        
        # Execute steps in sequence
        for step in self.steps:
            print(f"Running step: {step.name}")
            inputs = {
                input_name: step_outputs[dep.step_name][dep.output_name]
                for input_name, dep in step.input_dependencies.items()
                if dep.step_name in step_outputs
            }
            
            # Execute the step
            outputs = step.execute(inputs, config)
            step_outputs[step.name] = outputs
            
            # Store artifacts
            for output_name, output_value in outputs.items():
                artifact = Artifact(
                    name=f"{step.name}_{output_name}",
                    data=output_value,
                    metadata={
                        "step": step.name,
                        "output": output_name,
                        "pipeline": self.name,
                        "pipeline_uuid": self.uuid,
                        "created_at": datetime.datetime.now().isoformat()
                    }
                )
                artifacts[f"{step.name}_{output_name}"] = artifact
                
        return artifacts
    
    def save_config(self, path: str) -> None:
        """Save pipeline configuration to YAML file."""
        config = {
            "name": self.name,
            "description": self.description,
            "uuid": self.uuid,
            "metadata": self.metadata,
            "steps": [step.to_dict() for step in self.steps]
        }
        
        with open(path, 'w') as f:
            yaml.dump(config, f)
            
    @classmethod
    def load_config(cls, path: str) -> 'Pipeline':
        """Load pipeline configuration from YAML file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            
        pipeline = cls(name=config['name'], description=config['description'])
        pipeline.uuid = config['uuid']
        pipeline.metadata = config['metadata']
        
        # Load steps
        for step_config in config['steps']:
            step = Step.from_dict(step_config)
            pipeline.add_step(step)
            
        return pipeline


# core/step.py
from typing import Dict, Any, Optional, Callable, NamedTuple
import inspect
import hashlib
import json


class Dependency(NamedTuple):
    """Represents a dependency on another step's output."""
    step_name: str
    output_name: str


class Step:
    """Represents a single step in an ML pipeline."""
    
    def __init__(self, 
                 name: str,
                 function: Callable,
                 input_dependencies: Optional[Dict[str, Dependency]] = None,
                 description: Optional[str] = None,
                 cache_enabled: bool = True):
        self.name = name
        self.function = function
        self.input_dependencies = input_dependencies or {}
        self.description = description or ""
        self.cache_enabled = cache_enabled
        
        # Extract function signature to validate inputs/outputs
        self.signature = inspect.signature(function)
        
    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step function with provided inputs and configuration."""
        # Check for cached results if enabled
        if self.cache_enabled:
            cache_key = self._compute_cache_key(inputs, config)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                print(f"Using cached result for step {self.name}")
                return cached_result
        
        # Execute function
        result = self.function(**inputs, config=config)
        
        # Handle different return types
        if result is None:
            outputs = {}
        elif isinstance(result, dict):
            outputs = result
        else:
            outputs = {"output": result}
            
        # Store result in cache if enabled
        if self.cache_enabled:
            self._store_in_cache(cache_key, outputs)
            
        return outputs
    
    def _compute_cache_key(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Compute a cache key based on inputs, configuration, and function code."""
        # Create a dictionary with all the relevant data for caching
        cache_data = {
            "step_name": self.name,
            "inputs": self._serializable_dict(inputs),
            "config": self._serializable_dict(config),
            "function_code": inspect.getsource(self.function)
        }
        
        # Convert to JSON and hash
        data_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _serializable_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a dictionary to a JSON-serializable form."""
        result = {}
        for key, value in data.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif hasattr(value, '__dict__'):
                result[key] = str(value)
            else:
                # Attempt to use the value as is, but this may fail for complex objects
                result[key] = value
        return result
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve results from cache if available."""
        # This is a placeholder; a real implementation would use a proper cache system
        # For this example, we'll always return None (no cache hit)
        return None
    
    def _store_in_cache(self, cache_key: str, outputs: Dict[str, Any]) -> None:
        """Store results in cache."""
        # This is a placeholder; a real implementation would store in a proper cache system
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "function": self.function.__name__,
            "module": self.function.__module__,
            "input_dependencies": {
                input_name: {"step_name": dep.step_name, "output_name": dep.output_name}
                for input_name, dep in self.input_dependencies.items()
            },
            "cache_enabled": self.cache_enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Step':
        """Create a Step instance from a dictionary."""
        # In a real implementation, we'd need to dynamically import the module and function
        # For this example, we'll create a dummy function
        def dummy_function(*args, **kwargs):
            return {}
        
        input_dependencies = {
            input_name: Dependency(dep["step_name"], dep["output_name"])
            for input_name, dep in data["input_dependencies"].items()
        }
        
        return cls(
            name=data["name"],
            function=dummy_function,
            input_dependencies=input_dependencies,
            description=data["description"],
            cache_enabled=data["cache_enabled"]
        )


# core/config.py
import os
import yaml
from typing import Dict, Any, Optional


class Config:
    """Configuration management for the MLOps framework."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        self.config = {}
        
        if config_dict:
            self.config.update(config_dict)
            
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, path: str) -> None:
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            file_config = yaml.safe_load(f)
            if file_config:
                self.config.update(file_config)
    
    def save_to_file(self, path: str) -> None:
        """Save configuration to a YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.config, f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        # Support for nested keys using dot notation
        if '.' in key:
            parts = key.split('.')
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        # Support for nested keys using dot notation
        if '.' in key:
            parts = key.split('.')
            config = self.config
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            config[parts[-1]] = value
        else:
            self.config[key] = value
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)


# storage/artifact.py
import uuid
import datetime
import json
import os
import pickle
from typing import Any, Dict, Optional


class Artifact:
    """Represents data artifacts produced by pipeline steps."""
    
    def __init__(self, 
                 name: str,
                 data: Any,
                 metadata: Optional[Dict[str, Any]] = None,
                 artifact_id: Optional[str] = None):
        self.name = name
        self.data = data
        self.metadata = metadata or {}
        self.artifact_id = artifact_id or str(uuid.uuid4())
        
        # Add creation timestamp if not present
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = datetime.datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact metadata to dictionary (excluding data)."""
        return {
            "name": self.name,
            "artifact_id": self.artifact_id,
            "metadata": self.metadata,
            # We don't include the data as it might not be JSON serializable
        }
    
    def save(self, directory: str) -> str:
        """Save artifact to disk."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save metadata as JSON
        metadata_path = os.path.join(directory, f"{self.artifact_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.to_dict(), f)
        
        # Save data as pickle
        data_path = os.path.join(directory, f"{self.artifact_id}_data.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(self.data, f)
            
        return metadata_path
    
    @classmethod
    def load(cls, artifact_id: str, directory: str) -> 'Artifact':
        """Load artifact from disk."""
        # Load metadata
        metadata_path = os.path.join(directory, f"{artifact_id}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
            
        # Load data
        data_path = os.path.join(directory, f"{artifact_id}_data.pkl")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        return cls(
            name=metadata_dict["name"],
            data=data,
            metadata=metadata_dict["metadata"],
            artifact_id=metadata_dict["artifact_id"]
        )


# storage/local.py
import os
import json
import shutil
from typing import Dict, Any, List, Optional
from ..core.config import Config
from .artifact import Artifact


class LocalStorage:
    """Local filesystem storage for artifacts and metadata."""
    
    def __init__(self, base_directory: str):
        self.base_directory = base_directory
        
        # Create base directory if it doesn't exist
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)
            
        # Create directories for different storage types
        self.artifact_dir = os.path.join(base_directory, "artifacts")
        self.pipeline_dir = os.path.join(base_directory, "pipelines")
        self.config_dir = os.path.join(base_directory, "configs")
        
        for directory in [self.artifact_dir, self.pipeline_dir, self.config_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def save_artifact(self, artifact: Artifact) -> str:
        """Save an artifact to local storage."""
        # Create a directory for this artifact
        artifact_path = os.path.join(self.artifact_dir, artifact.artifact_id)
        if not os.path.exists(artifact_path):
            os.makedirs(artifact_path)
            
        return artifact.save(artifact_path)
    
    def load_artifact(self, artifact_id: str) -> Artifact:
        """Load an artifact from local storage."""
        artifact_path = os.path.join(self.artifact_dir, artifact_id)
        return Artifact.load(artifact_id, artifact_path)
    
    def list_artifacts(self, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List artifacts with optional metadata filtering."""
        artifacts = []
        
        for artifact_id in os.listdir(self.artifact_dir):
            metadata_path = os.path.join(self.artifact_dir, artifact_id, f"{artifact_id}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Apply filter if provided
                if filter_metadata:
                    matches = True
                    for key, value in filter_metadata.items():
                        if key not in metadata.get("metadata", {}) or metadata["metadata"][key] != value:
                            matches = False
                            break
                    
                    if not matches:
                        continue
                        
                artifacts.append(metadata)
                
        return artifacts
    
    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact from local storage."""
        artifact_path = os.path.join(self.artifact_dir, artifact_id)
        if os.path.exists(artifact_path):
            shutil.rmtree(artifact_path)
            return True
        return False
    
    def save_pipeline_config(self, pipeline_id: str, config: Dict[str, Any]) -> str:
        """Save a pipeline configuration to local storage."""
        pipeline_path = os.path.join(self.pipeline_dir, f"{pipeline_id}.json")
        with open(pipeline_path, 'w') as f:
            json.dump(config, f)
        return pipeline_path
    
    def load_pipeline_config(self, pipeline_id: str) -> Dict[str, Any]:
        """Load a pipeline configuration from local storage."""
        pipeline_path = os.path.join(self.pipeline_dir, f"{pipeline_id}.json")
        with open(pipeline_path, 'r') as f:
            return json.load(f)
    
    def save_framework_config(self, config: Config) -> str:
        """Save framework configuration to local storage."""
        config_path = os.path.join(self.config_dir, "mlops_config.yaml")
        config.save_to_file(config_path)
        return config_path
    
    def load_framework_config(self) -> Config:
        """Load framework configuration from local storage."""
        config_path = os.path.join(self.config_dir, "mlops_config.yaml")
        return Config(config_path=config_path)


# components/data_ingestion.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import os
import json


def csv_data_source(file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Data source component for loading CSV files."""
    separator = config.get('separator', ',')
    header = config.get('header', 0)
    
    df = pd.read_csv(file_path, sep=separator, header=header)
    
    return {
        "data": df,
        "metadata": {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "source_type": "csv",
            "source_path": file_path
        }
    }


def json_data_source(file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Data source component for loading JSON files."""
    orient = config.get('orient', 'records')
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    df = pd.json_normalize(data)
    
    return {
        "data": df,
        "metadata": {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "source_type": "json",
            "source_path": file_path
        }
    }


def database_source(connection_string: str, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Data source component for loading data from a database."""
    # In a real implementation, this would use SQLAlchemy or a similar library
    # For this example, we'll simulate a database query
    
    # Simulated data
    data = {
        "id": np.arange(100),
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "target": np.random.choice([0, 1], size=100)
    }
    df = pd.DataFrame(data)
    
    return {
        "data": df,
        "metadata": {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "source_type": "database",
            "query": query
        }
    }


def data_partitioner(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Component to partition data into training, validation, and test sets."""
    df = data["data"]
    test_ratio = config.get('test_ratio', 0.2)
    val_ratio = config.get('val_ratio', 0.2)
    random_seed = config.get('random_seed', 42)
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get indices for each partition
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    test_size = int(len(df) * test_ratio)
    val_size = int(len(df) * val_ratio)
    train_size = len(df) - test_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create partitions
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    return {
        "train_data": train_df,
        "val_data": val_df,
        "test_data": test_df,
        "metadata": {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "train_ratio": train_size / len(df),
            "val_ratio": val_size / len(df),
            "test_ratio": test_size / len(df),
            "random_seed": random_seed
        }
    }


# components/data_validation.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, validator, Field


class ValidationSchema(BaseModel):
    """Base schema for data validation."""
    pass


class NumericFeatureSchema(ValidationSchema):
    """Schema for numeric features."""
    name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_missing_percentage: float = 0.1
    
    @validator('allowed_missing_percentage')
    def validate_missing_percentage(cls, v):
        if v < 0 or v > 1:
            raise ValueError("allowed_missing_percentage must be between 0 and 1")
        return v


class CategoricalFeatureSchema(ValidationSchema):
    """Schema for categorical features."""
    name: str
    allowed_values: Optional[List[str]] = None
    allowed_missing_percentage: float = 0.1
    
    @validator('allowed_missing_percentage')
    def validate_missing_percentage(cls, v):
        if v < 0 or v > 1:
            raise ValueError("allowed_missing_percentage must be between 0 and 1")
        return v


class DatasetSchema(BaseModel):
    """Schema for entire dataset."""
    numeric_features: List[NumericFeatureSchema] = Field(default_factory=list)
    categorical_features: List[CategoricalFeatureSchema] = Field(default_factory=list)
    required_columns: List[str] = Field(default_factory=list)
    min_rows: Optional[int] = None


def validate_data(data: Dict[str, Any], schema: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data against a schema."""
    df = data["data"]
    
    # Convert schema dict to DatasetSchema
    dataset_schema = DatasetSchema(
        numeric_features=[NumericFeatureSchema(**f) for f in schema.get("numeric_features", [])],
        categorical_features=[CategoricalFeatureSchema(**f) for f in schema.get("categorical_features", [])],
        required_columns=schema.get("required_columns", []),
        min_rows=schema.get("min_rows")
    )
    
    validation_errors = []
    
    # Check required columns
    missing_columns = [col for col in dataset_schema.required_columns if col not in df.columns]
    if missing_columns:
        validation_errors.append(f"Missing required columns: {missing_columns}")
    
    # Check minimum rows
    if dataset_schema.min_rows and len(df) < dataset_schema.min_rows:
        validation_errors.append(f"Dataset has {len(df)} rows, minimum required is {dataset_schema.min_rows}")
    
    # Validate numeric features
    for feature in dataset_schema.numeric_features:
        if feature.name not in df.columns:
            validation_errors.append(f"Numeric feature {feature.name} not found in dataset")
            continue
        
        # Check data type
        if not pd.api.types.is_numeric_dtype(df[feature.name]):
            validation_errors.append(f"Feature {feature.name} is not numeric")
            continue
        
        # Check missing values
        missing_percentage = df[feature.name].isna().mean()
        if missing_percentage > feature.allowed_missing_percentage:
            validation_errors.append(
                f"Feature {feature.name} has {missing_percentage:.2%} missing values, "
                f"maximum allowed is {feature.allowed_missing_percentage:.2%}"
            )
        
        # Check range
        non_null_values = df[feature.name].dropna()
        if feature.min_value is not None and non_null_values.min() < feature.min_value:
            validation_errors.append(
                f"Feature {feature.name} has minimum value {non_null_values.min()}, "
                f"minimum allowed is {feature.min_value}"
            )
        if feature.max_value is not None and non_null_values.max() > feature.max_value:
            validation_errors.append(
                f"Feature {feature.name} has maximum value {non_null_values.max()}, "
                f"maximum allowed is {feature.max_value}"
            )
    
    # Validate categorical features
    for feature in dataset_schema.categorical_features:
        if feature.name not in df.columns:
            validation_errors.append(f"Categorical feature {feature.name} not found in dataset")
            continue
        
        # Check missing values
        missing_percentage = df[feature.name].isna().mean()
        if missing_percentage > feature.allowed_missing_percentage:
            validation_errors.append(
                f"Feature {feature.name} has {missing_percentage:.2%} missing values, "
                f"maximum allowed is {feature.allowed_missing_percentage:.2%}"
            )
        
        # Check allowed values
        if feature.allowed_values:
            invalid_values = set(df[feature.name].dropna().unique()) - set(feature.allowed_values)
            if invalid_values:
                validation_errors.append(
                    f"Feature {feature.name} has invalid values: {invalid_values}"
                )
    
    # Determine if validation passed
    validation_passed = len(validation_errors) == 0
    
    return {
        "validation_passed": validation_passed,
        "validation_errors": validation_errors,
        "data": df,
        "metadata": {
            "validation_schema": schema,
            "error_count": len(validation_errors)
        }
    }


def data_drift_detector(reference_data: Dict[str, Any], current_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Detect data drift between reference and current datasets."""
    reference_df = reference_data["data"]
    current_df = current_data["data"]
    
    # Ensure both datasets have the same columns
    common_columns = list(set(reference_df.columns) & set(current_df.columns))
    ref_df = reference_df[common_columns]
    curr_df = current_df[common_columns]
    
    drift_results = {}
    drift_detected = False
    
    for col in common_columns:
        # Skip non-numeric columns for simple statistics
        if not pd.api.types.is_numeric_dtype(ref_df[col]) or not pd.api.types.is_numeric_dtype(curr_df[col]):
            continue
        
        # Calculate basic statistics
        ref_mean = ref_df[col].mean()
        curr_mean = curr_df[col].mean()
        ref_std = ref_df[col].std()
        curr_std = curr_df[col].std()
        
        # Simple drift detection based on mean and standard deviation
        mean_change_pct = abs((curr_mean - ref_mean) / ref_mean) if ref_mean != 0 else float('inf')
        std_change_pct = abs((curr_std - ref_std) / ref_std) if ref_std != 0 else float('inf')
        
        # Threshold for drift detection
        drift_threshold = config.get('drift_threshold', 0.1)
        
        col_drift_detected = mean_change_pct > drift_threshold or std_change_pct > drift_threshold
        drift_detected |= col_drift_detected
        
        drift_results[col] = {
            "ref_mean": ref_mean,
            "curr_mean": curr_mean,
            "mean_change_pct": mean_change_pct,
            "ref_std": ref_std,
            "curr_std": curr_std,
            "std_change_pct": std_change_pct,
            "drift_detected": col_drift_detected
        }
    
    return {
        "drift_detected": drift_detected,
        "drift_results": drift_results,
        "metadata": {
            "reference_rows": len(ref_df),
            "current_rows": len(curr_df),
            "columns_analyzed": common_columns,
            "drift_threshold": config.get('drift_threshold', 0.1)
        }
    }


# components/preprocessing.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle


def feature_selector(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Select features for model training."""
    df = data["data"]
    
    # Get features to keep
    features_to_keep = config.get('features', [])
    target_column = config.get('target_column')
    
    if not features_