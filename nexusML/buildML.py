import logging
import os
import json
import yaml
import pickle
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Any, Optional, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import onnx
import onnxruntime
import tensorflow as tf
from scipy.stats import chisquare
from sklearn.metrics.pairwise import rbf_kernel
from azure.monitor import AzureMonitorClient
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v1.api.metrics_api import MetricsApi
import boto3
import prometheus_client

# Configure logging
logging.basicConfig(level=logging.INFO)

class PipelineStack:
    """Enhanced pipeline stack with all integrated features"""
    def __init__(self, name, artifact_root, steps: List['Step'], max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = name
        self.steps = steps
        self.artifact_store: Dict[str, Artifact] = {}
        self.artifact_root = artifact_root
        os.makedirs(artifact_root, exist_ok=True)
        self.serializer = ModelSerializer(os.path.join(artifact_root, "models"))
        self.drift_detector = None

    # Existing methods (run_tasks, save_pipeline, visualize_pipeline, get_artifact) remain the same
    # Add new integration points

    def initialize_drift_detector(self, reference_data: np.ndarray):
        """Initialize drift detection system with reference data"""
        self.drift_detector = DataDriftDetector(reference_data)
        self.artifact_store['drift_detector'] = Artifact(self.drift_detector)
        self.logger.info("Drift detector initialized with reference data")

    def get_cloud_monitor(self, config: dict) -> 'CloudMonitoringService':
        """Factory method for cloud monitoring services"""
        providers = {
            'aws': AWSCloudWatchMonitor,
            'prometheus': PrometheusMonitor,
            'azure': AzureMonitor,
            'datadog': DatadogMonitor
        }
        provider_class = providers.get(config.get('provider', 'prometheus'))
        return provider_class(config) if provider_class else None

class Config:
    """Enhanced configuration class with all settings"""
    def __init__(self, config_dict: Dict = None):
        self.config_dict = config_dict or {
            'serialization': {
                'formats': ['onnx', 'tensorflow', 'pickle'],
                'onnx_input_shape': [None, 4]
            },
            'monitoring': {
                'provider': 'prometheus',
                'azure_connection_string': '',
                'datadog_api_key': ''
            },
            'retraining': {
                'strategy': 'shadow_mode',
                'performance_threshold': 0.85,
                'drift_thresholds': {
                    'psi': 0.2,
                    'mmd': 0.1,
                    'chi_square': 0.05
                }
            }
        }

    # Existing methods remain the same

# class Artifact:
#     """Enhanced artifact class with versioning"""
#     def __init__(self, data, version: str = None):
#         self.data = data
#         self.version = version or datetime.now().strftime("%Y%m%d%H%M%S")
#         self.created_at = datetime.now()

class ModelSerializer:
    """Multi-format model serializer with fallback support"""
    def __init__(self, artifact_root: str):
        self.artifact_root = artifact_root
        os.makedirs(artifact_root, exist_ok=True)

    def serialize(self, model: Any, metadata: dict, config: dict) -> Dict[str, str]:
        """Serialize model in multiple formats with fallback"""
        results = {}
        for fmt in config.get('formats', ['pickle']):
            try:
                if fmt == 'onnx':
                    results[fmt] = self._serialize_onnx(model, metadata, config)
                elif fmt == 'tensorflow':
                    results[fmt] = self._serialize_tf(model, metadata)
                else:
                    results[fmt] = self._serialize_pickle(model, metadata)
            except Exception as e:
                logging.warning(f"Serialization failed for {fmt}: {str(e)}")
        return results

    def _serialize_onnx(self, model, metadata, config):
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        initial_type = [('float_input', FloatTensorType(config['onnx_input_shape']))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        filename = os.path.join(self.artifact_root, f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}.onnx")
        onnx.save(onnx_model, filename)
        return filename

    def _serialize_tf(self, model, metadata):
        filename = os.path.join(self.artifact_root, f"tf_model_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        tf.saved_model.save(model, filename)
        return filename

    def _serialize_pickle(self, model, metadata):
        filename = os.path.join(self.artifact_root, f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump({'model': model, 'metadata': metadata}, f)
        return filename

class DataDriftDetector:
    """Advanced drift detection system"""
    def __init__(self, reference_data: np.ndarray):
        self.reference_data = reference_data
        self.methods = {
            'psi': self._calculate_psi,
            'mmd': self._calculate_mmd,
            'chi_square': self._calculate_chi_square
        }

    def detect_drift(self, current_data: np.ndarray) -> Dict[str, float]:
        """Calculate drift scores using all methods"""
        return {method: fn(current_data) for method, fn in self.methods.items()}

    # Implement statistical methods (same as previous)

class CloudMonitoringService:
    """Unified cloud monitoring interface"""
    # Implementations for AWS, Azure, Datadog, Prometheus (same as previous)

class RetrainingOrchestrator:
    """Advanced retraining management system"""
    def __init__(self, config: dict):
        self.config = config
        self.performance_history = []
        self.drift_history = []
        self.active_models = {}

    def evaluate_retraining_needs(self, metrics: dict, drift_scores: dict) -> bool:
        """Determine if retraining is needed based on multiple factors"""
        conditions = [
            metrics['accuracy'] < self.config['performance_threshold'],
            drift_scores['psi'] > self.config['drift_thresholds']['psi'],
            drift_scores['mmd'] > self.config['drift_thresholds']['mmd'],
            drift_scores['chi_square'] > self.config['drift_thresholds']['chi_square']
        ]
        return any(conditions)

    # A/B testing and shadow mode methods (same as previous)

class TrainingPipeline:
    """Complete integrated training pipeline"""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = Config().config_dict
        self.pipeline_stack = PipelineStack(
            name="full_ml_pipeline",
            artifact_root="artifacts",
            steps=self._create_steps(),
            max_workers=4
        )
        self.pipeline_stack.artifact_store['config'] = Artifact(self.config)

    def _create_steps(self) -> List[Step]:
        """Define all pipeline steps"""
        return [
            Step(
                name="data_ingestion",
                inputs=["data_path", "config"],
                outputs=["raw_data"],
                task=self.data_ingestion_task,
                critical=True
            ),
            Step(
                name="data_validation",
                inputs=["raw_data", "config"],
                outputs=["validated_data"],
                task=self.data_validation_task
            ),
            Step(
                name="data_transformation",
                inputs=["validated_data", "config"],
                outputs=["processed_data"],
                task=self.data_transformation_task
            ),
            Step(
                name="model_training",
                inputs=["processed_data", "config"],
                outputs=["trained_model"],
                task=self.model_training_task,
                critical=True
            ),
            Step(
                name="model_evaluation",
                inputs=["trained_model", "processed_data", "config"],
                outputs=["model_metrics"],
                task=self.model_evaluation_task
            ),
            Step(
                name="model_serialization",
                inputs=["trained_model", "model_metrics", "config"],
                outputs=["serialized_models"],
                task=self.model_serialization_task
            ),
            Step(
                name="deployment",
                inputs=["serialized_models", "model_metrics", "config"],
                outputs=["deployment_status"],
                task=self.deployment_task,
                critical=True
            ),
            Step(
                name="monitoring_setup",
                inputs=["config", "processed_data"],
                outputs=["monitoring_system"],
                task=self.monitoring_setup_task
            ),
            Step(
                name="retraining_analysis",
                inputs=["monitoring_system", "model_metrics", "config"],
                outputs=["retraining_decision"],
                task=self.retraining_analysis_task
            )
        ]

    # Task implementations with integrated features

    def data_ingestion_task(self):
        self.logger.info("Ingesting data...")
        # Actual data loading logic here
        raw_data = np.random.rand(1000, 4)  # Placeholder
        self.pipeline_stack.initialize_drift_detector(raw_data[:, :3])
        return {"raw_data": raw_data}

    def model_serialization_task(self):
        model = self.pipeline_stack.get_artifact("trained_model")
        metrics = self.pipeline_stack.get_artifact("model_metrics")
        serialized = self.pipeline_stack.serializer.serialize(
            model.data,
            metrics.data,
            self.config['serialization']
        )
        return {"serialized_models": serialized}

    def deployment_task(self):
        self.logger.info("Deploying model...")
        models = self.pipeline_stack.get_artifact("serialized_models")
        metrics = self.pipeline_stack.get_artifact("model_metrics")
        
        orchestrator = RetrainingOrchestrator(self.config['retraining'])
        
        if self.config['retraining']['strategy'] == 'shadow_mode':
            deployment_status = orchestrator.shadow_mode_deployment(
                primary_model=models.data,
                shadow_model=None  # Would load previous model in real scenario
            )
        elif self.config['retraining']['strategy'] == 'ab_testing':
            deployment_status = orchestrator.a_b_testing_deployment(
                model_a=models.data,
                model_b=None  # Previous model version
            )
        
        return {"deployment_status": deployment_status}

    def monitoring_setup_task(self):
        self.logger.info("Initializing monitoring...")
        monitor = self.pipeline_stack.get_cloud_monitor(
            self.config['monitoring']
        )
        return {"monitoring_system": {
            'service': monitor,
            'drift_detector': self.pipeline_stack.drift_detector
        }}

    def retraining_analysis_task(self):
        self.logger.info("Analyzing retraining needs...")
        monitor = self.pipeline_stack.get_artifact("monitoring_system")
        metrics = self.pipeline_stack.get_artifact("model_metrics")
        current_data = self._get_production_data()  # Would implement real data fetch
        
        drift_scores = monitor.data['drift_detector'].detect_drift(current_data)
        orchestrator = RetrainingOrchestrator(self.config['retraining'])
        needs_retraining = orchestrator.evaluate_retraining_needs(
            metrics.data,
            drift_scores
        )
        
        return {"retraining_decision": {
            "required": needs_retraining,
            "drift_scores": drift_scores,
            "last_metrics": metrics.data
        }}

    # Other task methods and helper functions

    def run(self):
        """Execute full pipeline"""
        try:
            self.pipeline_stack.artifact_store["data_path"] = Artifact(self.data_path)
            self.pipeline_stack.run_tasks([step.task for step in self.pipeline_stack.steps])
            self.pipeline_stack.save_pipeline()
            self.pipeline_stack.visualize_pipeline()
            return True
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return False

# if __name__ == "__main__":
#     pipeline = TrainingPipeline("data/example.csv")
#     if pipeline.run():
#         print("Pipeline executed successfully!")
#         retraining_status = pipeline.pipeline_stack.get_artifact("retraining_decision")
#         if retraining_status.data['required']:
#             print("System requires retraining based on:")
#             print(json.dumps(retraining_status.data, indent=2))
#     else:
#         print("Pipeline execution failed")





# import logging
# import os
# import json
# import yaml
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import List, Dict, Callable, Any, Optional
# import matplotlib.pyplot as plt
# import networkx as nx

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# class PipelineStack:
#     """Base class for pipeline stacks with parallel task execution."""
#     def __init__(self, name, artifact_root, steps: List['Step'], max_workers: int = 4):
#         self.max_workers = max_workers
#         self.logger = logging.getLogger(self.__class__.__name__)
#         self.name = name
#         self.steps = steps
#         self.artifact_store: Dict[str, Artifact] = {}
#         self.artifact_root = artifact_root
#         os.makedirs(artifact_root, exist_ok=True)

#     def run_tasks(self, tasks: List[Callable]):
#         """Execute tasks in parallel using ThreadPoolExecutor."""
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = {executor.submit(task): task.__name__ for task in tasks}
#             results = {}
#             for future in as_completed(futures):
#                 task_name = futures[future]
#                 try:
#                     task_results = future.result()
#                     if task_results and isinstance(task_results, dict):
#                         # Store task results as artifacts
#                         for name, data in task_results.items():
#                             self.artifact_store[name] = Artifact(data)
#                             self.logger.info(f"Artifact '{name}' created and stored")
#                     results[task_name] = task_results
#                     self.logger.info(f"Task {task_name} completed successfully.")
#                 except Exception as e:
#                     self.logger.error(f"Task {task_name} failed: {str(e)}")
#                     # Propagate exception to stop pipeline if critical
#                     if task_name in [step.name for step in self.steps]:
#                         step = next(step for step in self.steps if step.name == task_name)
#                         if step.critical:
#                             raise RuntimeError(f"Critical step {task_name} failed: {str(e)}")
#             return results

#     def save_pipeline(self, filename: Optional[str] = None):
#         if filename is None:
#             filename = os.path.join(self.artifact_root, "pipeline_state.json")
        
#         os.makedirs(os.path.dirname(filename), exist_ok=True)
#         state = {
#             "pipeline_name": self.name,
#             "steps": [step.name for step in self.steps],
#             "artifacts": {
#                 name: (
#                     artifact.data.config_dict
#                     if isinstance(artifact.data, Config)
#                     else artifact.data
#                 )
#                 for name, artifact in self.artifact_store.items()
#                 if hasattr(artifact, 'data')  # Check if the artifact has data attribute
#             },
#         }
#         with open(filename, "w") as file:
#             json.dump(state, file, indent=4)
#         self.logger.info(f"Pipeline state saved to {filename}")

#     def visualize_pipeline(self, output_file: Optional[str] = None):
#         if output_file is None:
#             output_file = os.path.join(self.artifact_root, "pipeline_graph")
            
#         G = nx.DiGraph()
        
#         # Add nodes for steps with one color
#         step_nodes = [step.name for step in self.steps]
#         nx.add_nodes_from(step_nodes, node_type='step')
        
#         # Add nodes for artifacts with a different color
#         artifact_nodes = set()
#         for step in self.steps:
#             artifact_nodes.update(step.inputs)
#             artifact_nodes.update(step.outputs)
#         # Remove step names from artifact nodes
#         artifact_nodes = artifact_nodes - set(step_nodes)
#         nx.add_nodes_from(artifact_nodes, node_type='artifact')
        
#         # Add edges
#         for step in self.steps:
#             for input_name in step.inputs:
#                 G.add_edge(input_name, step.name)
#             for output_name in step.outputs:
#                 G.add_edge(step.name, output_name)

#         plt.figure(figsize=(12, 8))
#         pos = nx.spring_layout(G, seed=42)  # Layout algorithm for positioning
        
#         # Draw nodes with different colors based on type
#         node_colors = ['lightblue' if G.nodes[n].get('node_type') == 'step' else 'lightgreen' 
#                        for n in G.nodes()]
        
#         nx.draw(
#             G,
#             pos,
#             with_labels=True,
#             node_color=node_colors,
#             edge_color="gray",
#             node_size=2000,
#             font_size=10,
#             font_weight="bold",
#         )

#         # Add legend
#         plt.legend(["Steps", "Artifacts"], loc="upper right")
        
#         os.makedirs(os.path.dirname(output_file), exist_ok=True)
#         plt.savefig(f"{output_file}.png")
#         plt.close()  # Close the figure to free memory
#         self.logger.info(f"Pipeline visualization saved as {output_file}.png")

#     def get_artifact(self, name: str) -> Optional[Any]:
#         """Retrieve an artifact from the store by name."""
#         artifact = self.artifact_store.get(name)
#         if artifact:
#             return artifact.data
#         return None


# class Config:
#     def __init__(self, config_dict: Dict = None):
#         self.config_dict = config_dict or {}

#     def get(self, key: str, default: Any = None):
#         return self.config_dict.get(key, default)

#     @staticmethod
#     def load_from_file(filename: str):
#         """Loads configuration from a YAML or JSON file."""
#         try:
#             with open(filename, "r") as file:
#                 if filename.endswith((".yml", ".yaml")):
#                     config_data = yaml.safe_load(file)
#                 else:
#                     config_data = json.load(file)
#             return Config(config_data)
#         except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
#             raise ValueError(f"Error loading config file {filename}: {e}")


# class Artifact:
#     """Class for storing pipeline artifacts."""
#     def __init__(self, data):
#         self.data = data
#         self.created_at = None  # Could add timestamp functionality


# class Step:
#     """Represents a step in the pipeline."""
#     def __init__(self, name: str, inputs: List[str], outputs: List[str], task: Callable, critical: bool = False):
#         self.name = name
#         self.inputs = inputs
#         self.outputs = outputs
#         self.task = task
#         self.critical = critical  # Flag to indicate if step failure should stop pipeline
#         # Store the task name as the function name
#         self.task.__name__ = name


# class TrainingPipeline:
#     def __init__(self, path: str):
#         self.path = path
#         self.logger = logging.getLogger(self.__class__.__name__)
#         try:
#             self.config = Config.load_from_file("config/config.yml")
#         except ValueError as e:
#             self.logger.warning(f"Config file not found or invalid: {e}")
#             self.logger.info("Using default configuration")
#             self.config = Config({
#                 "data_split_ratio": 0.8,
#                 "random_seed": 42,
#                 "features": ["feature1", "feature2", "feature3"],
#                 "target": "target_column"
#             })

#     def run(self):
#         # Define steps
#         data_ingestion_step = Step(
#             name="data_ingestion",
#             inputs=["path", "config"],
#             outputs=["train_data", "test_data"],
#             task=self.data_ingestion_task,
#             critical=True  # Mark as critical - pipeline will stop if this fails
#         )
#         data_validation_step = Step(
#             name="data_validation",
#             inputs=["train_data", "test_data", "config"],
#             outputs=["val_train", "val_test"],
#             task=self.data_validation_task
#         )
#         data_transformation_step = Step(
#             name="data_transformation",
#             inputs=["val_train", "val_test", "config"],
#             outputs=["X_train", "X_test", "y_train", "y_test"],
#             task=self.data_transformation_task
#         )

#         steps = [data_ingestion_step, data_validation_step, data_transformation_step]

#         # Create stack with steps
#         self.pipeline_stack = PipelineStack(
#             name="ml_customer_churn",
#             artifact_root="artifacts",
#             steps=steps,
#             max_workers=2
#         )
        
#         # Add path to artifact store
#         self.pipeline_stack.artifact_store["path"] = Artifact(self.path)
#         self.pipeline_stack.artifact_store["config"] = Artifact(self.config)

#         try:
#             # Run tasks
#             self.pipeline_stack.run_tasks([step.task for step in steps])

#             # Save and visualize pipeline
#             self.pipeline_stack.save_pipeline()
#             self.pipeline_stack.visualize_pipeline()
            
#             self.logger.info("Pipeline completed successfully")
#             return True
#         except Exception as e:
#             self.logger.error(f"Pipeline failed: {str(e)}")
#             return False

#     def data_ingestion_task(self):
#         self.logger.info("Running data ingestion task...")
#         # Here you would actually load data from self.path
#         # For now using placeholder logic
#         return {"train_data": "placeholder_train_data", "test_data": "placeholder_test_data"}

#     def data_validation_task(self):
#         self.logger.info("Running data validation task...")
#         # In a real implementation, you would:
#         # 1. Get the train_data and test_data from the artifact store
#         # train_data = self.pipeline_stack.get_artifact("train_data")
#         # test_data = self.pipeline_stack.get_artifact("test_data")
#         # 2. Perform validation
#         # 3. Return validated data
#         return {"val_train": "validated_train_data", "val_test": "validated_test_data"}

#     def data_transformation_task(self):
#         self.logger.info("Running data transformation task...")
#         # Similarly, would get validated data and transform it
#         return {
#             "X_train": "transformed_X_train", 
#             "X_test": "transformed_X_test", 
#             "y_train": "transformed_y_train", 
#             "y_test": "transformed_y_test"
#         }


# if __name__ == "__main__":
#     path = "data/churn-train.csv"
#     pipe_instance = TrainingPipeline(path)
#     pipe_instance.run()