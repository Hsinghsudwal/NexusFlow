from .artifacts_Config import *
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

# 🔹 Step Class to Define Pipeline Steps
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

        logging.info(f"Step '{self.name}' initialized with function: {self.function.__name__}")

    def execute(self, context: Context):
        """Execute the step with the provided input artifacts."""
        start_time = datetime.datetime.now()
        logging.info(f"Executing step: {self.name}")

        # Extract inputs from context
        input_data = [context.get(input_name).data for input_name in self.inputs if context.get(input_name)]

        try:
            result = self.function(*input_data, **self.config)
        except Exception as e:
            logging.error(f"Error executing step {self.name}: {e}")
            raise

        self.execution_time = (datetime.datetime.now() - start_time).total_seconds()
        logging.info(f"Step {self.name} completed in {self.execution_time:.2f}s")

        self.executed = True

        # Save outputs to context as Artifacts
        for output_name in self.outputs:
            context.update(output_name, Artifact(name=output_name, data=result))

        return result

# 🔹 Pipeline Class to Manage and Execute Steps
class Pipeline:
    def __init__(self, name: str, config: Config, max_workers: int = 2):
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

        logging.info(f"Executing pipeline '{self.name}' in parallel mode.")
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_step = {executor.submit(step.execute, context): step for step in self.steps}

            for future in as_completed(future_to_step):
                step = future_to_step[future]
                try:
                    result = future.result()
                    self.execution_metadata[step.name] = {
                        "execution_time": step.execution_time,
                        "inputs": step.inputs,
                        "outputs": step.outputs
                    }
                except Exception as e:
                    logging.error(f"Step {step.name} failed: {e}")

        return context

    def save_pipeline(self, filename: str = "pipeline_state.json"):
        """Save pipeline metadata and results to a JSON file."""
        with open(filename, "w") as file:
            json.dump(self.execution_metadata, file, indent=4)
        logging.info(f"Pipeline state saved to {filename}")

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
        logging.info(f"Pipeline visualization saved as {output_file}.png")
