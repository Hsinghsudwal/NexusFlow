"""
Custom MLOps Framework

A lightweight framework for managing ML pipelines with configurable execution modes,
artifact tracking, and model stacking capabilities.
"""

import os
import json
import time
import uuid
import pickle
import logging
import datetime
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
from dataclasses import dataclass, field


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Core components
@dataclass
class Artifact:
    """Representation of an output from a pipeline step."""
    name: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    uri: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    def save(self, storage_path: str) -> str:
        """Save artifact to disk."""
        os.makedirs(storage_path, exist_ok=True)
        file_path = os.path.join(storage_path, f"{self.name}_{self.id}.pkl")
        
        with open(file_path, 'wb') as f:
            pickle.dump(self.data, f)
        
        metadata_path = os.path.join(storage_path, f"{self.name}_{self.id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "name": self.name,
                "id": self.id,
                "created_at": self.created_at,
                "metadata": self.metadata,
                "uri": file_path
            }, f, indent=2)
            
        self.uri = file_path
        return file_path
    
    @classmethod
    def load(cls, artifact_id: str, storage_path: str) -> 'Artifact':
        """Load artifact from disk."""
        metadata_files = [f for f in os.listdir(storage_path) if f.endswith('_metadata.json')]
        for meta_file in metadata_files:
            with open(os.path.join(storage_path, meta_file), 'r') as f:
                metadata = json.load(f)
                if metadata['id'] == artifact_id:
                    with open(metadata['uri'], 'rb') as data_file:
                        data = pickle.load(data_file)
                    return cls(
                        name=metadata['name'],
                        data=data,
                        metadata=metadata['metadata'],
                        uri=metadata['uri'],
                        id=metadata['id'],
                        created_at=metadata['created_at']
                    )
        raise ValueError(f"Artifact with ID {artifact_id} not found in {storage_path}")


class Step(ABC):
    """Base class for pipeline steps."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.upstream_steps: List[Step] = []
        self.upstream_artifacts: Dict[str, Artifact] = {}
        self.output_artifact: Optional[Artifact] = None
    
    def connect(self, upstream_step: 'Step') -> 'Step':
        """Connect this step to an upstream step."""
        self.upstream_steps.append(upstream_step)
        return self
    
    def set_upstream_artifact(self, step_name: str, artifact: Artifact) -> None:
        """Set an upstream artifact for this step."""
        self.upstream_artifacts[step_name] = artifact
    
    @abstractmethod
    def execute(self, context: Dict[str, Any] = None) -> Artifact:
        """Execute the step and return an artifact."""
        pass
    
    def __call__(self, context: Dict[str, Any] = None) -> Artifact:
        """Make the step callable."""
        logger.info(f"Executing step: {self.name}")
        start_time = time.time()
        
        # Create context if none provided
        if context is None:
            context = {}
        
        # Add upstream artifacts to context
        context['upstream_artifacts'] = self.upstream_artifacts
        
        # Execute step
        self.output_artifact = self.execute(context)
        
        # Log execution time
        execution_time = time.time() - start_time
        logger.info(f"Step {self.name} completed in {execution_time:.2f} seconds")
        
        return self.output_artifact


class Pipeline:
    """Pipeline for organizing and executing steps."""
    
    def __init__(self, name: str, storage_path: str = './artifacts'):
        self.name = name
        self.steps: Dict[str, Step] = {}
        self.storage_path = storage_path
        self.artifacts: Dict[str, Artifact] = {}
        self.run_id: str = str(uuid.uuid4())
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.datetime.now().isoformat(),
            'status': 'created'
        }
        os.makedirs(storage_path, exist_ok=True)
    
    def add_step(self, step: Step) -> 'Pipeline':
        """Add a step to the pipeline."""
        self.steps[step.name] = step
        return self
    
    def run(self, parallel: bool = False, context: Dict[str, Any] = None) -> Dict[str, Artifact]:
        """Run the pipeline with all steps."""
        logger.info(f"Running pipeline: {self.name} (Run ID: {self.run_id})")
        self.metadata['status'] = 'running'
        self.metadata['start_time'] = datetime.datetime.now().isoformat()
        
        # Create run-specific storage directory
        run_storage_path = os.path.join(self.storage_path, self.run_id)
        os.makedirs(run_storage_path, exist_ok=True)
        
        # Save pipeline metadata
        with open(os.path.join(run_storage_path, 'pipeline_metadata.json'), 'w') as f:
            json.dump({
                'name': self.name,
                'run_id': self.run_id,
                'metadata': self.metadata,
                'steps': list(self.steps.keys())
            }, f, indent=2)
        
        # Determine execution order (topological sort)
        execution_order = self._get_execution_order()
        
        if context is None:
            context = {}
        
        # Store global context for the pipeline run
        context['pipeline'] = {
            'name': self.name,
            'run_id': self.run_id,
            'storage_path': run_storage_path
        }
        
        if parallel:
            self._run_parallel(execution_order, context, run_storage_path)
        else:
            self._run_sequential(execution_order, context, run_storage_path)
        
        self.metadata['status'] = 'completed'
        self.metadata['end_time'] = datetime.datetime.now().isoformat()
        
        # Update pipeline metadata
        with open(os.path.join(run_storage_path, 'pipeline_metadata.json'), 'w') as f:
            json.dump({
                'name': self.name,
                'run_id': self.run_id,
                'metadata': self.metadata,
                'steps': list(self.steps.keys()),
                'artifacts': {name: artifact.id for name, artifact in self.artifacts.items()}
            }, f, indent=2)
        
        return self.artifacts
    
    def _run_sequential(self, execution_order: List[str], context: Dict[str, Any], storage_path: str) -> None:
        """Run pipeline steps sequentially."""
        logger.info("Running pipeline in sequential mode")
        
        for step_name in execution_order:
            step = self.steps[step_name]
            
            # Make sure upstream artifacts are available
            for upstream_step in step.upstream_steps:
                if upstream_step.name in self.artifacts:
                    step.set_upstream_artifact(upstream_step.name, self.artifacts[upstream_step.name])
            
            # Execute step
            artifact = step(context)
            artifact.save(storage_path)
            
            # Store artifact
            self.artifacts[step_name] = artifact
    
    def _run_parallel(self, execution_order: List[str], context: Dict[str, Any], storage_path: str) -> None:
        """Run pipeline steps in parallel where possible."""
        logger.info("Running pipeline in parallel mode")
        
        # Group steps by their level in the pipeline (steps that can run in parallel)
        levels = self._group_steps_by_level()
        
        for level_steps in levels:
            # Create futures for all steps in this level
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = {}
                
                for step_name in level_steps:
                    step = self.steps[step_name]
                    
                    # Make sure upstream artifacts are available
                    for upstream_step in step.upstream_steps:
                        if upstream_step.name in self.artifacts:
                            step.set_upstream_artifact(upstream_step.name, self.artifacts[upstream_step.name])
                    
                    # Submit step for execution
                    futures[step_name] = executor.submit(step, context)
                
                # Wait for all steps in this level to complete
                for step_name, future in futures.items():
                    artifact = future.result()
                    artifact.save(storage_path)
                    self.artifacts[step_name] = artifact
    
    def _get_execution_order(self) -> List[str]:
        """Determine the topological ordering of steps."""
        result = []
        visited = set()
        temp = set()
        
        def visit(step_name):
            if step_name in temp:
                raise ValueError(f"Circular dependency detected in pipeline: {step_name}")
            if step_name in visited:
                return
            
            temp.add(step_name)
            step = self.steps[step_name]
            
            for upstream_step in step.upstream_steps:
                visit(upstream_step.name)
            
            temp.remove(step_name)
            visited.add(step_name)
            result.append(step_name)
        
        for step_name in self.steps:
            if step_name not in visited:
                visit(step_name)
        
        # Reverse to get correct execution order
        result.reverse()
        return result
    
    def _group_steps_by_level(self) -> List[List[str]]:
        """Group steps by their level in the pipeline (for parallel execution)."""
        # Calculate dependencies for each step
        dependencies = {step_name: set() for step_name in self.steps}
        for step_name, step in self.steps.items():
            for upstream_step in step.upstream_steps:
                dependencies[step_name].add(upstream_step.name)
        
        levels = []
        remaining_steps = set(self.steps.keys())
        
        while remaining_steps:
            # Find steps with no remaining dependencies
            current_level = [step_name for step_name in remaining_steps 
                            if not dependencies[step_name]]
            
            if not current_level:
                raise ValueError("Circular dependency detected in pipeline")
            
            levels.append(current_level)
            remaining_steps -= set(current_level)
            
            # Remove completed steps from dependencies
            for step_name in remaining_steps:
                dependencies[step_name] -= set(current_level)
        
        return levels


# Stack implementation
class StackingPipeline(Pipeline):
    """Pipeline that supports model stacking."""
    
    def __init__(self, name: str, storage_path: str = './artifacts'):
        super().__init__(name, storage_path)
        self.models: List[Step] = []
        self.preprocessors: List[Step] = []
        self.meta_model: Optional[Step] = None
    
    def add_base_model(self, model_step: Step, preprocessor_step: Optional[Step] = None) -> 'StackingPipeline':
        """Add a base model and its preprocessor to the stacking ensemble."""
        self.add_step(model_step)
        self.models.append(model_step)
        
        if preprocessor_step:
            self.add_step(preprocessor_step)
            self.preprocessors.append(preprocessor_step)
            model_step.connect(preprocessor_step)
        
        return self
    
    def set_meta_model(self, meta_model_step: Step) -> 'StackingPipeline':
        """Set the meta-model for stacking."""
        self.add_step(meta_model_step)
        self.meta_model = meta_model_step
        
        # Connect meta-model to all base models
        for model_step in self.models:
            self.meta_model.connect(model_step)
        
        return self


# Standard step implementations
class DataLoaderStep(Step):
    """Step for loading data."""
    
    def __init__(self, name: str = "DataLoader", data_loader_fn: Callable = None):
        super().__init__(name)
        self.data_loader_fn = data_loader_fn
    
    def execute(self, context: Dict[str, Any] = None) -> Artifact:
        """Load data and return as artifact."""
        if self.data_loader_fn:
            data = self.data_loader_fn(context)
        else:
            # Default implementation if no loader function provided
            data = context.get('data', None)
            if data is None:
                raise ValueError("No data provided in context and no data_loader_fn specified")
        
        return Artifact(name=f"{self.name}_output", data=data)


class PreprocessorStep(Step):
    """Step for preprocessing data."""
    
    def __init__(self, name: str = "Preprocessor", preprocessor_fn: Callable = None):
        super().__init__(name)
        self.preprocessor_fn = preprocessor_fn
    
    def execute(self, context: Dict[str, Any] = None) -> Artifact:
        """Preprocess data and return processed data as artifact."""
        # Get data from upstream artifact or context
        upstream_artifacts = context.get('upstream_artifacts', {})
        if not upstream_artifacts and 'data' not in context:
            raise ValueError("No upstream artifacts or data provided in context")
        
        # Use the first upstream artifact as input if available
        if upstream_artifacts:
            input_data = next(iter(upstream_artifacts.values())).data
        else:
            input_data = context['data']
        
        # Apply preprocessing
        if self.preprocessor_fn:
            processed_data = self.preprocessor_fn(input_data, context)
        else:
            # Default implementation
            processed_data = input_data
        
        return Artifact(name=f"{self.name}_output", data=processed_data)


class ModelTrainerStep(Step):
    """Step for training a model."""
    
    def __init__(self, 
                 name: str = "ModelTrainer", 
                 model_class: Any = None, 
                 model_params: Dict[str, Any] = None,
                 train_fn: Callable = None):
        super().__init__(name)
        self.model_class = model_class
        self.model_params = model_params or {}
        self.train_fn = train_fn
        self.model = None
    
    def execute(self, context: Dict[str, Any] = None) -> Artifact:
        """Train model and return it as artifact."""
        # Get data from upstream artifact or context
        upstream_artifacts = context.get('upstream_artifacts', {})
        if not upstream_artifacts and 'data' not in context:
            raise ValueError("No upstream artifacts or data provided in context")
        
        # Use the first upstream artifact as input if available
        if upstream_artifacts:
            input_data = next(iter(upstream_artifacts.values())).data
        else:
            input_data = context['data']
        
        # If custom train function is provided, use it
        if self.train_fn:
            model = self.train_fn(input_data, self.model_params, context)
        else:
            # Default implementation - expects X_train, y_train in input_data
            if not isinstance(input_data, dict) or 'X_train' not in input_data or 'y_train' not in input_data:
                raise ValueError("Input data must contain 'X_train' and 'y_train' keys")
            
            if self.model_class is None:
                raise ValueError("Model class must be provided if no train_fn is specified")
            
            model = self.model_class(**self.model_params)
            model.fit(input_data['X_train'], input_data['y_train'])
        
        self.model = model
        
        # Create artifact with model and validation metrics if available
        artifact = Artifact(name=f"{self.name}_model", data=model)
        
        # Add validation metrics if available
        if isinstance(input_data, dict) and 'X_val' in input_data and 'y_val' in input_data and hasattr(model, 'score'):
            try:
                val_score = model.score(input_data['X_val'], input_data['y_val'])
                artifact.metadata['validation_score'] = val_score
            except Exception as e:
                logger.warning(f"Could not calculate validation score: {e}")
        
        return artifact


class PredictorStep(Step):
    """Step for making predictions with a trained model."""
    
    def __init__(self, name: str = "Predictor", predict_fn: Callable = None):
        super().__init__(name)
        self.predict_fn = predict_fn
    
    def execute(self, context: Dict[str, Any] = None) -> Artifact:
        """Make predictions and return them as artifact."""
        upstream_artifacts = context.get('upstream_artifacts', {})
        if not upstream_artifacts:
            raise ValueError("No upstream artifacts provided")
        
        # Find model artifact and data artifact
        model_artifact = None
        data_artifact = None
        
        for step_name, artifact in upstream_artifacts.items():
            if '_model' in artifact.name:
                model_artifact = artifact
            elif 'data' in artifact.name or 'preprocessor' in artifact.name:
                data_artifact = artifact
        
        if model_artifact is None:
            raise ValueError("No model artifact found in upstream artifacts")
        
        if data_artifact is None:
            if 'data' not in context:
                raise ValueError("No data artifact found in upstream artifacts and no data in context")
            data = context['data']
        else:
            data = data_artifact.data
        
        # Get model
        model = model_artifact.data
        
        # Make predictions
        if self.predict_fn:
            predictions = self.predict_fn(model, data, context)
        else:
            # Default implementation
            if isinstance(data, dict) and 'X_test' in data:
                predictions = model.predict(data['X_test'])
            else:
                predictions = model.predict(data)
        
        return Artifact(name=f"{self.name}_predictions", data=predictions)


class EvaluatorStep(Step):
    """Step for evaluating model predictions."""
    
    def __init__(self, name: str = "Evaluator", metrics: Dict[str, Callable] = None, evaluate_fn: Callable = None):
        super().__init__(name)
        self.metrics = metrics or {}
        self.evaluate_fn = evaluate_fn
    
    def execute(self, context: Dict[str, Any] = None) -> Artifact:
        """Evaluate predictions and return metrics as artifact."""
        upstream_artifacts = context.get('upstream_artifacts', {})
        if not upstream_artifacts:
            raise ValueError("No upstream artifacts provided")
        
        # Find predictions artifact and true values
        predictions_artifact = None
        data_artifact = None
        
        for step_name, artifact in upstream_artifacts.items():
            if 'predictions' in artifact.name:
                predictions_artifact = artifact
            elif 'data' in artifact.name:
                data_artifact = artifact
        
        if predictions_artifact is None:
            raise ValueError("No predictions artifact found in upstream artifacts")
        
        predictions = predictions_artifact.data
        
        # Get true values
        if data_artifact is not None and isinstance(data_artifact.data, dict) and 'y_test' in data_artifact.data:
            true_values = data_artifact.data['y_test']
        elif 'y_test' in context:
            true_values = context['y_test']
        else:
            raise ValueError("No true values found in upstream artifacts or context")
        
        # Evaluate predictions
        if self.evaluate_fn:
            metrics = self.evaluate_fn(predictions, true_values, context)
        else:
            # Default implementation using provided metrics
            metrics = {}
            for metric_name, metric_fn in self.metrics.items():
                metrics[metric_name] = metric_fn(true_values, predictions)
        
        return Artifact(name=f"{self.name}_metrics", data=metrics)


class StackingStep(Step):
    """Step for combining predictions from multiple models."""
    
    def __init__(self, name: str = "Stacker", stacking_fn: Callable = None, meta_model_class: Any = None, meta_model_params: Dict[str, Any] = None):
        super().__init__(name)
        self.stacking_fn = stacking_fn
        self.meta_model_class = meta_model_class
        self.meta_model_params = meta_model_params or {}
        self.meta_model = None
    
    def execute(self, context: Dict[str, Any] = None) -> Artifact:
        """Combine predictions from multiple models and return stacked model as artifact."""
        upstream_artifacts = context.get('upstream_artifacts', {})
        if not upstream_artifacts:
            raise ValueError("No upstream artifacts provided")
        
        # Collect base model predictions
        base_models = []
        base_predictions = []
        
        for step_name, artifact in upstream_artifacts.items():
            if '_model' in artifact.name:
                base_models.append(artifact.data)
            elif 'predictions' in artifact.name:
                base_predictions.append(artifact.data)
        
        if not base_models:
            raise ValueError("No base models found in upstream artifacts")
        
        # If we don't have predictions, we need to get data to make predictions
        if not base_predictions:
            if 'data' not in context:
                raise ValueError("No base predictions or data found")
            
            # Make predictions with base models
            data = context['data']
            for model in base_models:
                if isinstance(data, dict) and 'X_train' in data:
                    preds = model.predict(data['X_train'])
                else:
                    preds = model.predict(data)
                base_predictions.append(preds)
        
        # Stack models
        if self.stacking_fn:
            meta_model = self.stacking_fn(base_models, base_predictions, context)
        else:
            # Default implementation
            if 'y_train' not in context and not isinstance(context.get('data', {}), dict):
                raise ValueError("Target values not found in context")
            
            y_train = context.get('y_train', context.get('data', {}).get('y_train'))
            
            # Stack predictions into a single feature matrix
            import numpy as np
            X_meta = np.column_stack(base_predictions)
            
            # Train meta-model
            meta_model = self.meta_model_class(**self.meta_model_params)
            meta_model.fit(X_meta, y_train)
        
        self.meta_model = meta_model
        
        return Artifact(name=f"{self.name}_stacked_model", data={
            'meta_model': meta_model,
            'base_models': base_models
        })


# Utility functions
def save_pipeline(pipeline: Pipeline, path: str) -> str:
    """Save pipeline configuration to disk."""
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{pipeline.name}.pkl")
    
    with open(file_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    return file_path


def load_pipeline(file_path: str) -> Pipeline:
    """Load pipeline configuration from disk."""
    with open(file_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    return pipeline
