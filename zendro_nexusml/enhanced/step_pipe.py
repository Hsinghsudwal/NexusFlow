import os
import json
import logging
import datetime
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union, Set
from dataclasses import dataclass
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Step definition (equivalent to a pipeline node)
@dataclass
class Step:
    """A step represents a single processing unit in a pipeline."""
    name: str
    func: Callable
    inputs: List[str]
    outputs: List[str]
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

# Pipeline definition
class Pipeline:
    """A pipeline is a DAG of steps to be executed."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps = []
        self.step_names = set()
        
    def add_step(self, step: Step) -> None:
        """Add a step to the pipeline."""
        if step.name in self.step_names:
            raise ValueError(f"Step with name '{step.name}' already exists in pipeline")
        
        self.steps.append(step)
        self.step_names.add(step.name)
        
    def get_step_by_name(self, name: str) -> Optional[Step]:
        """Get a step by its name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None
        
    def visualize(self) -> None:
        """Print a simple visualization of the pipeline structure."""
        print(f"Pipeline: {self.name}")
        print(f"Description: {self.description}")
        print("Steps:")
        
        for step in self.steps:
            print(f"  - {step.name}")
            print(f"    Inputs: {', '.join(step.inputs) if step.inputs else 'None'}")
            print(f"    Outputs: {', '.join(step.outputs)}")
            
    def save(self, filepath: str) -> None:
        """Save pipeline definition to YAML file."""
        pipeline_dict = {
            "name": self.name,
            "description": self.description,
            "steps": [
                {
                    "name": step.name,
                    "inputs": step.inputs,
                    "outputs": step.outputs,
                    "parameters": step.parameters
                }
                for step in self.steps
            ]
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(pipeline_dict, f, default_flow_style=False)
        
        logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, step_funcs: Dict[str, Callable]) -> 'Pipeline':
        """Load pipeline definition from YAML file."""
        with open(filepath, 'r') as f:
            pipeline_dict = yaml.safe_load(f)
        
        pipeline = cls(name=pipeline_dict["name"], description=pipeline_dict.get("description", ""))
        
        for step_dict in pipeline_dict["steps"]:
            step_name = step_dict["name"]
            if step_name not in step_funcs:
                raise ValueError(f"Function for step '{step_name}' not provided")
                
            step = Step(
                name=step_name,
                func=step_funcs[step_name],
                inputs=step_dict["inputs"],
                outputs=step_dict["outputs"],
                parameters=step_dict.get("parameters", {})
            )
            pipeline.add_step(step)
            
        return pipeline

# Data manager for handling pipeline data
class DataManager:
    """Manages data artifacts for pipeline steps."""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        self.artifacts = {}
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
    def register_artifact(self, name: str, filepath: Optional[str] = None, 
                    description: str = "", file_format: str = "pickle") -> None:
        """Register an artifact with the data manager."""
        if filepath is None:
            filepath = self.base_dir / f"{name}.{file_format}"
        else:
            filepath = Path(filepath)
            
        self.artifacts[name] = {
            "filepath": filepath,
            "description": description,
            "format": file_format
        }
        
    def save(self, name: str, data: Any) -> None:
        """Save data artifact to storage."""
        if name not in self.artifacts:
            raise ValueError(f"Artifact '{name}' not registered in data manager")
        
        artifact_info = self.artifacts[name]
        filepath = artifact_info["filepath"]
        file_format = artifact_info["format"]
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if file_format == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif file_format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Saved artifact '{name}' to {filepath}")
        
    def load(self, name: str) -> Any:
        """Load data artifact from storage."""
        if name not in self.artifacts:
            raise ValueError(f"Artifact '{name}' not registered in data manager")
        
        artifact_info = self.artifacts[name]
        filepath = artifact_info["filepath"]
        file_format = artifact_info["format"]
        
        if not filepath.exists():
            raise FileNotFoundError(f"Artifact file does not exist: {filepath}")
        
        if file_format == "pickle":
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        elif file_format == "json":
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Loaded artifact '{name}' from {filepath}")
        return data

# Execution context for running pipelines
class ExecutionContext:
    """Context for executing pipeline steps, handling data and parameters."""
    
    def __init__(self, pipeline: Pipeline, data_manager: DataManager, 
                 parameters: Dict[str, Any] = None):
        self.pipeline = pipeline
        self.data_manager = data_manager
        self.parameters = parameters or {}
        self.artifacts = {}
        
    def execute_step(self, step: Step) -> Dict[str, Any]:
        """Execute a single step."""
        logger.info(f"Executing step: {step.name}")
        
        # Prepare step inputs
        inputs = {}
        for input_name in step.inputs:
            if input_name in self.artifacts:
                inputs[input_name] = self.artifacts[input_name]
            else:
                try:
                    inputs[input_name] = self.data_manager.load(input_name)
                except (ValueError, FileNotFoundError) as e:
                    logger.error(f"Failed to load input '{input_name}' for step '{step.name}': {e}")
                    raise
        
        # Prepare step parameters (combine global and step-specific)
        step_params = {**self.parameters, **step.parameters}
        
        # Execute step function
        try:
            outputs = step.func(inputs, step_params)
        except Exception as e:
            logger.error(f"Error executing step '{step.name}': {e}")
            raise
        
        # Process outputs
        if isinstance(outputs, dict):
            for output_name, output_data in outputs.items():
                if output_name in step.outputs:
                    self.artifacts[output_name] = output_data
                    try:
                        self.data_manager.save(output_name, output_data)
                    except ValueError as e:
                        logger.warning(f"Could not save output '{output_name}': {e}")
        else:
            # If there's just one output, use the first output name
            if step.outputs:
                output_name = step.outputs[0]
                self.artifacts[output_name] = outputs
                try:
                    self.data_manager.save(output_name, outputs)
                except ValueError as e:
                    logger.warning(f"Could not save output '{output_name}': {e}")
        
        logger.info(f"Step '{step.name}' executed successfully")
        return {name: self.artifacts.get(name) for name in step.outputs}

# Runner for executing pipelines
class PipelineRunner:
    """Responsible for executing pipelines with parallelization."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers
        
    def _build_dependency_graph(self, pipeline: Pipeline) -> Dict[str, Set[str]]:
        """Build a graph of step dependencies."""
        # For each step, which steps need to be completed before it can run
        dependencies = {step.name: set() for step in pipeline.steps}
        
        # For each output, which step produces it
        output_producers = {}
        for step in pipeline.steps:
            for output in step.outputs:
                output_producers[output] = step.name
                
        # Build dependencies
        for step in pipeline.steps:
            for input_name in step.inputs:
                if input_name in output_producers:
                    dependencies[step.name].add(output_producers[input_name])
                    
        return dependencies
    
    def _find_ready_steps(self, pipeline: Pipeline, dependencies: Dict[str, Set[str]], 
                         completed_steps: Set[str]) -> List[Step]:
        """Find steps that are ready to execute (all dependencies satisfied)."""
        ready_steps = []
        
        for step in pipeline.steps:
            if step.name in completed_steps:
                continue
                
            # Check if all dependencies are completed
            if dependencies[step.name].issubset(completed_steps):
                ready_steps.append(step)
                
        return ready_steps
    
    def run(self, context: ExecutionContext) -> Dict[str, Any]:
        """Run the pipeline with parallel execution where possible."""
        pipeline = context.pipeline
        completed_steps = set()
        
        # Build dependency graph
        dependencies = self._build_dependency_graph(pipeline)
        
        # Keep track of run status
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        while len(completed_steps) < len(pipeline.steps):
            ready_steps = self._find_ready_steps(pipeline, dependencies, completed_steps)
            
            if not ready_steps:
                remaining_steps = [s.name for s in pipeline.steps if s.name not in completed_steps]
                raise RuntimeError(f"Pipeline execution stalled. Remaining steps: {remaining_steps}")
            
            # Execute steps in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create tasks for each ready step
                futures = {
                    executor.submit(context.execute_step, step): step.name
                    for step in ready_steps
                }
                
                # Process completed tasks
                for future in futures:
                    step_name = futures[future]
                    try:
                        _ = future.result()  # Get results and handle any exceptions
                        completed_steps.add(step_name)
                    except Exception as e:
                        logger.error(f"Step '{step_name}' failed: {e}")
                        raise
        
        logger.info(f"Pipeline '{pipeline.name}' execution completed successfully (Run ID: {run_id})")
        return context.artifacts

# Main MLOps orchestrator
class MLOps:
    """Main orchestrator for the MLOps framework."""
    
    def __init__(self, project_dir: str = "./mlops_project"):
        self.project_dir = Path(project_dir)
        self.pipelines = {}
        self.data_manager = DataManager(base_dir=self.project_dir / "data")
        self.runner = PipelineRunner()
        
        # Create project structure
        (self.project_dir / "data").mkdir(exist_ok=True, parents=True)
        (self.project_dir / "pipelines").mkdir(exist_ok=True, parents=True)
        (self.project_dir / "logs").mkdir(exist_ok=True, parents=True)
        
    def set_max_workers(self, max_workers: int) -> None:
        """Set maximum number of worker threads."""
        self.runner.max_workers = max_workers
        
    def register_pipeline(self, pipeline: Pipeline) -> None:
        """Register a pipeline with the framework."""
        self.pipelines[pipeline.name] = pipeline
        
        # Auto-register artifacts based on pipeline
        for step in pipeline.steps:
            for output in step.outputs:
                if output not in self.data_manager.artifacts:
                    self.data_manager.register_artifact(output)
        
    def run_pipeline(self, pipeline_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a registered pipeline."""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not registered")
        
        pipeline = self.pipelines[pipeline_name]
        
        # Create execution context
        context = ExecutionContext(
            pipeline=pipeline,
            data_manager=self.data_manager,
            parameters=parameters or {}
        )
        
        # Run the pipeline
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.project_dir / "logs" / run_id
        run_dir.mkdir(exist_ok=True)
        
        logger.info(f"Starting pipeline run '{pipeline_name}' with ID: {run_id}")
        
        try:
            results = self.runner.run(context)
            
            # Save run metadata
            run_metadata = {
                "run_id": run_id,
                "pipeline": pipeline_name,
                "parameters": parameters or {},
                "start_time": datetime.datetime.now().isoformat(),
                "status": "completed"
            }
            
            with open(run_dir / "metadata.json", 'w') as f:
                json.dump(run_metadata, f)
                
            logger.info(f"Pipeline run '{pipeline_name}' completed successfully")
            return results
            
        except Exception as e:
            # Save failure metadata
            run_metadata = {
                "run_id": run_id,
                "pipeline": pipeline_name,
                "parameters": parameters or {},
                "start_time": datetime.datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            }
            
            with open(run_dir / "metadata.json", 'w') as f:
                json.dump(run_metadata, f)
                
            logger.error(f"Pipeline run '{pipeline_name}' failed: {e}")
            raise

# Example usage in main function
def main():
    """Example main function showing how to use the MLOps framework."""
    
    # Define step functions
    def load_data(inputs, params):
        """Load data step."""
        print(f"Loading data with params: {params}")
        # In a real scenario, this would load from a file or database
        data = {
            "features": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0]
        }
        return {"raw_data": data}
    
    def process_data(inputs, params):
        """Process data step."""
        print(f"Processing data with params: {params}")
        raw_data = inputs["raw_data"]
        
        # Simple transform - multiply features by scale factor
        scale_factor = params.get("scale_factor", 1.0)
        processed_data = {
            "features": [x * scale_factor for x in raw_data["features"]],
            "target": raw_data["target"]
        }
        
        return {"processed_data": processed_data}
    
    def split_data(inputs, params):
        """Split data into train/test sets."""
        print(f"Splitting data with params: {params}")
        data = inputs["processed_data"]
        
        # Simple split - first 70% for training
        split_idx = int(len(data["features"]) * 0.7)
        
        train_data = {
            "features": data["features"][:split_idx],
            "target": data["target"][:split_idx]
        }
        
        test_data = {
            "features": data["features"][split_idx:],
            "target": data["target"][split_idx:]
        }
        
        return {
            "train_data": train_data,
            "test_data": test_data
        }
    
    def compute_stats(inputs, params):
        """Compute statistics on the data."""
        print(f"Computing stats with params: {params}")
        train_data = inputs["train_data"]
        test_data = inputs["test_data"]
        
        stats = {
            "train_size": len(train_data["features"]),
            "test_size": len(test_data["features"]),
            "train_feature_mean": sum(train_data["features"]) / len(train_data["features"]),
            "test_feature_mean": sum(test_data["features"]) / len(test_data["features"])
        }
        
        return {"statistics": stats}
    
    # Initialize MLOps framework
    mlops = MLOps(project_dir="./example_project")
    mlops.set_max_workers(4)  # Set maximum parallel workers
    
    # Create a pipeline
    pipeline = Pipeline(
        name="data_pipeline",
        description="Example pipeline for data processing"
    )
    
    # Add steps to the pipeline
    pipeline.add_step(Step(
        name="load_data",
        func=load_data,
        inputs=[],
        outputs=["raw_data"]
    ))
    
    pipeline.add_step(Step(
        name="process_data",
        func=process_data,
        inputs=["raw_data"],
        outputs=["processed_data"],
        parameters={"scale_factor": 2.0}
    ))
    
    pipeline.add_step(Step(
        name="split_data",
        func=split_data,
        inputs=["processed_data"],
        outputs=["train_data", "test_data"]
    ))
    
    pipeline.add_step(Step(
        name="compute_stats",
        func=compute_stats,
        inputs=["train_data", "test_data"],
        outputs=["statistics"]
    ))
    
    # Register the pipeline
    mlops.register_pipeline(pipeline)
    
    # Visualize the pipeline
    pipeline.visualize()
    
    # Run the pipeline
    parameters = {"scale_factor": 3.0}  # Override the default scale_factor
    results = mlops.run_pipeline("data_pipeline", parameters=parameters)
    
    # Print results
    print("\nResults from pipeline execution:")
    print(f"Statistics: {results.get('statistics', {})}")
    
    # Save pipeline definition
    pipeline.save("./example_project/pipelines/data_pipeline.yaml")
    
    return results

if __name__ == "__main__":
    main()
