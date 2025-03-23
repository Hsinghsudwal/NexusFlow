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
from sklearn.metrics import accuracy_score, classification_report

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


# 🔹 Pipeline Class
class Pipeline:
    def __init__(self, name: str, config: Config, max_workers: int = 2):
        """
        Initialize the pipeline.
        """
        self.name = name
        self.config = config
        self.steps = []
        self.max_workers = max_workers
        self.execution_metadata = {}

    def add_step(self, step: Step):
        self.steps.append(step)

    def execute(self, execution_context: Dict):
        """Execute all steps in the pipeline using the context to store artifacts."""
        context = Context()

        # Initialize execution context
        for key, value in execution_context.items():
            context.update(key, Artifact(name=key, data=value))

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


# 🔹 Define Steps
def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate model performance."""
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    metrics = {
        'accuracy': accuracy,
        'classification_report': report
    }

    return predictions, metrics


def analyze_features(model, X_data):
    """Analyze feature importance."""
    feature_importance = model.feature_importances_
    return feature_importance


# 🔹 Define Pipeline
def create_pipeline():
    """Create and return the ML pipeline."""
    config = Config.load_from_file("config.json")
    pipeline = Pipeline(name="classification", config=config, max_workers=5)

    # Add steps to the pipeline
    pipeline.add_step(Step(name="evaluate_model", function=evaluate_model, inputs=["model", "X_test_scaled", "y_test"], outputs=["predictions", "metrics"]))
    pipeline.add_step(Step(name="analyze_features", function=analyze_features, inputs=["model", "X_data"], outputs=["feature_importance"]))

    return pipeline


# 🔹 Run the Pipeline
if __name__ == '__main__':
    pipeline = create_pipeline()

    execution_context = {
        "raw_data": "sample_data"  # Placeholder for actual data
    }
    
    pipeline.execute(execution_context)
    pipeline.save_pipeline()
    pipeline.visualize_pipeline()
    register_pipeline(pipeline)







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
from sklearn.metrics import accuracy_score, classification_report

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


# 🔹 Pipeline Class
class Pipeline:
    def __init__(self, name: str, config: Config, max_workers):
        """
        Initialize the pipeline.
        """
        self.name = name
        self.config = config
        self.steps = []
        self.max_workers = max_workers
        self.execution_metadata = {}

    def add_step(self, step: Step):
        self.steps.append(step)

    def execute(self, execution_context: Dict):
        """Execute all steps in the pipeline using the context to store artifacts."""
        context = Context()

        # Initialize execution context
        for key, value in execution_context.items():
            context.update(key, Artifact(name=key, data=value))

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


# 🔹 Define Steps
def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate model performance."""
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    metrics = {
        'accuracy': accuracy,
        'classification_report': report
    }

    return predictions, metrics


def analyze_features(model, X_data):
    """Analyze feature importance."""
    feature_importance = model.feature_importances_
    return feature_importance


# 🔹 Define Pipeline
def create_pipeline():
    """Create and return the ML pipeline."""
    config = Config.load_from_file("config.json")
    pipeline = Pipeline(name="classification", config=config, max_workers=5)

    # Add steps to the pipeline
    pipeline.add_step(Step(name="evaluate_model", function=evaluate_model, inputs=["model", "X_test_scaled", "y_test"], outputs=["predictions", "metrics"]))
    pipeline.add_step(Step(name="analyze_features", function=analyze_features, inputs=["model", "X_data"], outputs=["feature_importance"]))

    return pipeline


# 🔹 Run the Pipeline
if __name__ == '__main__':
    pipeline = create_pipeline()

    execution_context = {
        "raw_data": "sample_data"  # Placeholder for actual data
    }
    
    pipeline.execute(execution_context)
    pipeline.save_pipeline()
    pipeline.visualize_pipeline()
    register_pipeline(pipeline)


