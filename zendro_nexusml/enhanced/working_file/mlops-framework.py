import os
import json
import time
import logging
import datetime
from typing import Dict, List, Callable, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mlops-framework')

# Data Structures
@dataclass
class Artifact:
    """Represents data passed between pipeline steps."""
    name: str
    data: Any
    metadata: Dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    def save(self, directory: str):
        """Save artifact metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(directory, f"{self.name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'name': self.name,
                'metadata': self.metadata,
                'created_at': self.created_at
            }, f, indent=2)
        
        # Note: actual data saving should be implemented by child classes
        logger.info(f"Saved artifact metadata for '{self.name}' to {metadata_path}")
        
    @classmethod
    def load(cls, directory: str, name: str):
        """Load artifact metadata from disk."""
        metadata_path = os.path.join(directory, f"{name}_metadata.json")
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        # Note: actual data loading should be implemented by child classes
        return cls(name=data['name'], data=None, metadata=data['metadata'], created_at=data['created_at'])

@dataclass
class PipelineStep:
    """Represents a step in a pipeline."""
    name: str
    func: Callable
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    params: Dict = field(default_factory=dict)
    retry_count: int = 0
    timeout: Optional[int] = None

@dataclass
class PipelineRun:
    """Tracks a pipeline execution."""
    id: str
    pipeline_name: str
    start_time: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    end_time: Optional[str] = None
    status: str = "RUNNING"
    step_statuses: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, Dict] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        """Convert run to dictionary."""
        return asdict(self)
    
    def save(self, directory: str):
        """Save run metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        run_path = os.path.join(directory, f"run_{self.id}.json")
        with open(run_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved run metadata to {run_path}")
    
    @classmethod
    def load(cls, directory: str, run_id: str):
        """Load run metadata from disk."""
        run_path = os.path.join(directory, f"run_{run_id}.json")
        with open(run_path, 'r') as f:
            data = json.load(f)
        return cls(**data)

# Decorators for tracking and metrics
def track_time(func):
    """Decorator to track execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.2f} seconds")
        
        # If result is an Artifact or list of Artifacts, add execution time to metadata
        if isinstance(result, Artifact):
            result.metadata['execution_time'] = execution_time
        elif isinstance(result, list) and all(isinstance(item, Artifact) for item in result):
            for item in result:
                item.metadata['execution_time'] = execution_time
                
        return result
    return wrapper

def log_params(func):
    """Decorator to log parameters of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Function '{func.__name__}' called with kwargs: {kwargs}")
        return func(*args, **kwargs)
    return wrapper

# Main Pipeline Components
class Pipeline:
    """Main pipeline class for orchestrating steps."""
    
    def __init__(self, name: str, steps: List[PipelineStep] = None):
        self.name = name
        self.steps = steps or []
        self.artifacts_dir = "artifacts"
        self.runs_dir = "runs"
        self.max_workers = 4
        
    def add_step(self, step: PipelineStep):
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self
    
    def validate(self):
        """Validate the pipeline configuration."""
        # Check if all required inputs are available from previous steps
        all_outputs = set()
        for step in self.steps:
            # Check if all inputs for this step are provided by previous steps
            for input_name in step.inputs:
                if input_name not in all_outputs:
                    logger.warning(f"Step '{step.name}' requires input '{input_name}' which is not produced by any previous step")
            
            # Add outputs of this step to the set of all outputs
            all_outputs.update(step.outputs)
        
        # Check for duplicate step names
        step_names = [step.name for step in self.steps]
        if len(step_names) != len(set(step_names)):
            raise ValueError("Duplicate step names found in pipeline")
            
        logger.info(f"Pipeline '{self.name}' validated with {len(self.steps)} steps")
        return True
    
    def execute_step(self, step: PipelineStep, artifacts: Dict[str, Artifact], run: PipelineRun) -> Dict[str, Artifact]:
        """Execute a single pipeline step."""
        logger.info(f"Executing step: {step.name}")
        run.step_statuses[step.name] = "RUNNING"
        
        try:
            # Prepare inputs
            inputs = {}
            for input_name in step.inputs:
                if input_name in artifacts:
                    inputs[input_name] = artifacts[input_name].data
                else:
                    raise ValueError(f"Required input '{input_name}' not found for step '{step.name}'")
            
            # Execute the step function with inputs and parameters
            start_time = time.time()
            results = step.func(**inputs, **step.params)
            execution_time = time.time() - start_time
            
            # Update run metrics
            run.metrics[f"{step.name}_execution_time"] = execution_time
            
            # Process outputs
            if not isinstance(results, tuple) and len(step.outputs) == 1:
                # Single output
                output_name = step.outputs[0]
                artifacts[output_name] = Artifact(
                    name=output_name,
                    data=results,
                    metadata={"step": step.name, "execution_time": execution_time}
                )
                run.artifacts[output_name] = {"created_by": step.name}
            else:
                # Multiple outputs
                if len(results) != len(step.outputs):
                    raise ValueError(f"Step '{step.name}' returned {len(results)} outputs but expected {len(step.outputs)}")
                
                for i, output_name in enumerate(step.outputs):
                    artifacts[output_name] = Artifact(
                        name=output_name,
                        data=results[i],
                        metadata={"step": step.name, "execution_time": execution_time}
                    )
                    run.artifacts[output_name] = {"created_by": step.name}
            
            run.step_statuses[step.name] = "COMPLETED"
            logger.info(f"Step '{step.name}' completed successfully")
            return artifacts
            
        except Exception as e:
            run.step_statuses[step.name] = "FAILED"
            logger.error(f"Step '{step.name}' failed with error: {str(e)}")
            raise
    
    def run(self, params: Dict[str, Any] = None) -> PipelineRun:
        """Execute the pipeline."""
        self.validate()
        
        # Create run ID and initialize run
        run_id = f"{self.name}_{int(time.time())}"
        run = PipelineRun(id=run_id, pipeline_name=self.name)
        
        # Store parameters
        run.parameters = params or {}
        
        # Prepare artifact storage
        artifacts = {}
        
        try:
            # Identify steps that can be executed in parallel
            step_dependencies = {}
            for step in self.steps:
                step_dependencies[step.name] = set()
                for input_name in step.inputs:
                    for other_step in self.steps:
                        if input_name in other_step.outputs:
                            step_dependencies[step.name].add(other_step.name)
            
            # Execute steps in topological order, with parallelization where possible
            remaining_steps = self.steps.copy()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                while remaining_steps:
                    # Find steps with all dependencies satisfied
                    executable_steps = []
                    for step in remaining_steps:
                        if all(dep_name not in [s.name for s in remaining_steps] or 
                               run.step_statuses.get(dep_name) == "COMPLETED" 
                               for dep_name in step_dependencies[step.name]):
                            executable_steps.append(step)
                    
                    if not executable_steps:
                        raise ValueError("Circular dependency detected in pipeline")
                    
                    # Submit executable steps to thread pool
                    futures = {
                        executor.submit(self.execute_step, step, artifacts.copy(), run): step
                        for step in executable_steps
                    }
                    
                    # Process results as they complete
                    for future in futures:
                        step = futures[future]
                        try:
                            step_artifacts = future.result()
                            # Update artifacts with results from this step
                            artifacts.update(step_artifacts)
                        except Exception as e:
                            run.status = "FAILED"
                            logger.error(f"Pipeline execution failed at step '{step.name}': {str(e)}")
                            raise
                        
                        # Remove executed step from remaining steps
                        remaining_steps.remove(step)
            
            # Save artifacts
            for name, artifact in artifacts.items():
                artifact.save(self.artifacts_dir)
            
            # Update and save run metadata
            run.end_time = datetime.datetime.now().isoformat()
            run.status = "COMPLETED"
            run.save(self.runs_dir)
            
            logger.info(f"Pipeline '{self.name}' completed successfully")
            return run
            
        except Exception as e:
            run.end_time = datetime.datetime.now().isoformat()
            run.status = "FAILED"
            run.save(self.runs_dir)
            logger.error(f"Pipeline '{self.name}' failed: {str(e)}")
            raise

# Example usage
def main():
    """Example main function demonstrating the framework."""
    # Define some example pipeline steps
    @track_time
    @log_params
    def load_data(file_path: str):
        """Example step: Load data from a file."""
        logger.info(f"Loading data from {file_path}")
        # Simulate loading data
        time.sleep(1)
        data = {"example": "data"}
        return data
    
    @track_time
    @log_params
    def preprocess_data(data):
        """Example step: Preprocess the data."""
        logger.info("Preprocessing data")
        # Simulate preprocessing
        time.sleep(1.5)
        # Add a preprocessing step
        processed_data = data.copy()
        processed_data["preprocessed"] = True
        return processed_data
    
    @track_time
    @log_params
    def feature_engineering(data):
        """Example step: Extract features."""
        logger.info("Extracting features")
        # Simulate feature extraction
        time.sleep(2)
        features = {"features": [1, 2, 3]}
        return features, data
    
    # Create pipeline
    pipeline = Pipeline(name="example_pipeline")
    
    # Add steps
    pipeline.add_step(PipelineStep(
        name="load_data",
        func=load_data,
        inputs=[],
        outputs=["raw_data"],
        params={"file_path": "example.csv"}
    ))
    
    pipeline.add_step(PipelineStep(
        name="preprocess_data",
        func=preprocess_data,
        inputs=["raw_data"],
        outputs=["processed_data"],
        params={}
    ))
    
    pipeline.add_step(PipelineStep(
        name="feature_engineering",
        func=feature_engineering,
        inputs=["processed_data"],
        outputs=["features", "processed_data_with_features"],
        params={}
    ))
    
    # Run pipeline
    try:
        run = pipeline.run()
        logger.info(f"Pipeline run completed with status: {run.status}")
        logger.info(f"Run metrics: {run.metrics}")
    except Exception as e:
        logger.error(f"Pipeline run failed: {str(e)}")

if __name__ == "__main__":
    main()
