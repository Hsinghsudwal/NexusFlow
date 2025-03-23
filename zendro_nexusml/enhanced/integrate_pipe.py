import yaml
import json
import time
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import graphviz

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 🔹 Config Class for Loading and Managing Pipeline Configuration
class Config:
    def __init__(self, config_dict=None):
        self.config_dict = config_dict or {}

    def get(self, key, default=None):
        return self.config_dict.get(key, default)

    @staticmethod
    def load_from_file(filename):
        with open(filename, 'r') as file:
            config_data = json.load(file)
        return Config(config_data)


# 🔹 Artifact Class to Manage Step Outputs
class Artifact:
    def __init__(self, name, data=None):
        self.name = name
        self.data = data

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f)

    def load(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __repr__(self):
        return f"Artifact(name={self.name}, data={self.data})"


# 🔹 Context Class to Handle Artifacts During Execution
class Context:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def update(self, key, artifact: Artifact):
        self.data[key] = artifact
        logger.info(f"Artifact saved: {artifact.name}, data: {artifact.data}")



class Step:
    def __init__(self, name: str, function: Callable, description: Optional[str] = None,
                 inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None, config: Optional[Dict] = None):
        if function is None:
            raise ValueError(f"Function for step '{name}' cannot be None")
        
        self.name = name
        self.function = function
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.config = config or {}
        self.description = description
        self.executed = False
        self.execution_time = None

        logger.info(f"Step '{self.name}' initialized with function: {self.function.__name__}")

    def execute(self, context: Context):
        """Execute the step with the provided input artifacts."""
        if self.function is None:
            raise ValueError(f"No function assigned to step: {self.name}")

        start_time = datetime.now()
        logger.info(f"Executing step: {self.name}")

        # Extract inputs from context
        input_data = [context.get(input_name).data for input_name in self.inputs if context.get(input_name)]

        try:
            result = self.function(*input_data, **self.config)
        except Exception as e:
            logger.error(f"Error executing step {self.name}: {e}")
            raise

        self.execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Step {self.name} completed in {self.execution_time:.2f}s")

        self.executed = True

        # Save outputs to context as Artifacts
        for output_name in self.outputs:
            context.update(output_name, Artifact(name=output_name, data=result))

        return result

class Pipeline:
    def __init__(self, name: str, parallel: bool = False, max_workers: int = 2):
        """
        Initialize the pipeline.
        :param name: Name of the pipeline.
        :param parallel: Whether to execute steps in parallel.
        :param max_workers: Number of workers for parallel execution.
        """
        self.name = name
        self.steps = []
        self.parallel = parallel
        self.max_workers = max_workers
        self.execution_metadata = {}

    def add_step(self, step: Step):
        self.steps.append(step)

    def execute(self, context: Dict):
        """Execute all steps in the pipeline either sequentially or in parallel."""
        results = {}

        if self.parallel:
            logger.info(f"Executing pipeline '{self.name}' in parallel mode with {self.max_workers} workers.")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_step = {executor.submit(step.execute, results, context): step for step in self.steps}

                for future in as_completed(future_to_step):
                    step = future_to_step[future]
                    try:
                        results[step.name] = future.result()
                        self.execution_metadata[step.name] = {
                            "execution_time": step.execution_time,
                            "inputs": step.inputs,
                            "outputs": step.outputs
                        }
                    except Exception as e:
                        logger.error(f"Step {step.name} failed with error: {e}")
                        raise
        else:
            logger.info(f"Executing pipeline '{self.name}' in sequential mode.")
            for step in self.steps:
                results[step.name] = step.execute(results, context)
                self.execution_metadata[step.name] = {
                    "execution_time": step.execution_time,
                    "inputs": step.inputs,
                    "outputs": step.outputs
                }

        return results

    def get_pipeline_metadata(self):
        """Gather metadata about the pipeline execution."""
        return {
            "pipeline_name": self.name,
            "execution_metadata": self.execution_metadata
        }

    def save_pipeline(self, filename: str = "pipeline_state.json"):
        """Save pipeline metadata and results to a JSON file."""
        with open(filename, "w") as file:
            json.dump(self.get_pipeline_metadata(), file, indent=4)
        logger.info(f"Pipeline state saved to {filename}")

    def visualize_pipeline(self, output_file: str = "pipeline_graph"):
        """Generate a DAG visualization of the pipeline using Graphviz."""
        dot = graphviz.Digraph(comment=self.name)

        for step in self.steps:
            dot.node(step.name, step.name)

            for input_name in step.inputs:
                dot.edge(input_name, step.name)

            for output_name in step.outputs:
                dot.edge(step.name, output_name)

        dot.render(output_file, format="png", cleanup=True)
        logger.info(f"Pipeline visualization saved as {output_file}.png")


# **Pipeline Registration**
def register_pipeline(pipeline: Pipeline, registry_path: str = "registered_pipelines.pkl"):
    """Register the pipeline by saving its metadata to a file."""
    if os.path.exists(registry_path):
        with open(registry_path, "rb") as file:
            registry = pickle.load(file)
    else:
        registry = {}

    registry[pipeline.name] = pipeline.get_pipeline_metadata()

    with open(registry_path, "wb") as file:
        pickle.dump(registry, file)

    logger.info(f"Pipeline '{pipeline.name}' registered successfully.")


# Sample Step Functions
class DataProcessing:
    @staticmethod
    def preprocess_data(data: str, scale_factor: float = 1.0) -> str:
        """Mock preprocessing step"""
        logger.info(f"Preprocessing data with scale_factor={scale_factor}")
        time.sleep(1)  # Simulate delay
        return f"{data}_processed"

    @staticmethod
    def train_model(train_data: str, batch_size: int = 16) -> Dict[str, Any]:
        """Mock model training step"""
        logger.info(f"Training model on {train_data} with batch_size={batch_size}")
        time.sleep(2)
        return {"model_accuracy": 0.85, "loss": 0.15}


# **Main Execution**
if __name__ == "__main__":
    config = load_config("config.yaml")

    pipeline = Pipeline(config["pipeline_name"], parallel=True, max_workers=config["max_workers"])

    data_preprocessing_step = Step(
        name="DataPreprocessing",
        function=DataProcessing.preprocess_data,
        inputs=["raw_data"],
        outputs=["processed_data"],
        config=config.get("parameters", {})
    )

    training_step = Step(
        name="ModelTraining",
        function=DataProcessing.train_model,
        inputs=["processed_data"],
        outputs=["model_metrics"],
        config=config.get("parameters", {})
    )

    # Add Steps
    pipeline.add_step(data_preprocessing_step)
    pipeline.add_step(training_step)

    # Execute Pipeline
    execution_context = {"raw_data": "raw_sample_data"}
    results = pipeline.execute(execution_context)

    # Save Pipeline State & Metadata
    pipeline.save_pipeline()
    pipeline.visualize_pipeline()
    register_pipeline(pipeline)

    # Print Results
    logger.info("Pipeline Execution Completed.")
    logger.info(f"Final Results: {results}")
















# 🔹 Step Class


    


# 🔹 Pipeline Class
class Pipeline:
    def __init__(self, name: str, config: Config, use_thread_pool: bool = False, max_workers: int = 2):
        """
        Initialize the pipeline.
        :param name: Name of the pipeline.
        :param config: Config object containing pipeline settings.
        :param use_thread_pool: If True, use ThreadPoolExecutor for parallel execution.
        :param max_workers: Number of workers for parallel execution.
        """
        self.name = name
        self.config = config
        self.steps = []
        self.use_thread_pool = use_thread_pool
        self.max_workers = max_workers
        self.execution_metadata = {}

    def add_step(self, step: Step):
        self.steps.append(step)

    def execute(self, execution_context: Dict):
        """Execute all steps in the pipeline using the context to store artifacts."""
        context = Context()

        # Initialize the execution context in the Context class
        for key, value in execution_context.items():
            context.update(key, Artifact(name=key, data=value))

        if self.use_thread_pool:
            logger.info(f"Executing pipeline '{self.name}' in parallel mode with {self.max_workers} workers.")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_step = {executor.submit(step.execute, context): step for step in self.steps}

                for future in as_completed(future_to_step):
                    step = future_to_step[future]
                    try:
                        future.result()
                        self.execution_metadata[step.name] = {
                            "execution_time": step.execution_time,
                            "inputs": step.inputs,
                            "outputs": step.outputs
                        }
                    except Exception as e:
                        logger.error(f"Step {step.name} failed with error: {e}")
                        raise
        else:
            logger.info(f"Executing pipeline '{self.name}' in sequential mode.")
            for step in self.steps:
                step.execute(context)
                self.execution_metadata[step.name] = {
                    "execution_time": step.execution_time,
                    "inputs": step.inputs,
                    "outputs": step.outputs
                }

        return context

    def save_pipeline(self, filename: str = "pipeline_state.json"):
        """Save pipeline metadata and results to a JSON file."""
        with open(filename, "w") as file:
            json.dump(self.execution_metadata, file, indent=4)
        logger.info(f"Pipeline state saved to {filename}")

    def visualize_pipeline(self, output_file: str = "pipeline_graph"):
        """Generate a DAG visualization of the pipeline using Graphviz."""
        dot = graphviz.Digraph(comment=self.name)

        for step in self.steps:
            dot.node(step.name, step.name)

            for input_name in step.inputs:
                dot.edge(input_name, step.name)

            for output_name in step.outputs:
                dot.edge(step.name, output_name)

        dot.render(output_file, format="png", cleanup=True)
        logger.info(f"Pipeline visualization saved as {output_file}.png")


# **Pipeline Registration**
def register_pipeline(pipeline: Pipeline, registry_path: str = "registered_pipelines.pkl"):
    """Register the pipeline by saving its metadata to a file."""
    if os.path.exists(registry_path):
        with open(registry_path, "rb") as file:
            registry = pickle.load(file)
    else:
        registry = {}

    registry[pipeline.name] = pipeline.execution_metadata

    with open(registry_path, "wb") as file:
        pickle.dump(registry, file)

    logger.info(f"Pipeline '{pipeline.name}' registered successfully.")


# **Main Execution**
if __name__ == "__main__":
    config = Config.load_from_file("config.json")

    pipeline = Pipeline(config.get("pipeline_name"), config, use_thread_pool=config.get("use_thread_pool", False))

    # Sample Steps
    data_preprocessing_step = Step(
        name="DataPreprocessing",
        function=lambda data: data + "_processed",
        inputs=["raw_data"],
        outputs=["processed_data"]
    )

    training_step = Step(
        name="ModelTraining",
        function=lambda data: {"accuracy": 0.9},
        inputs=["processed_data"],
        outputs=["model_metrics"]
    )

    pipeline.add_step(data_preprocessing_step)
    pipeline.add_step(training_step)

    execution_context = {"raw_data": "sample_data"}
    pipeline.execute(execution_context)

    pipeline.save_pipeline()
    pipeline.visualize_pipeline()
    register_pipeline(pipeline)
