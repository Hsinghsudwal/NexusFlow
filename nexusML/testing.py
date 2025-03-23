import logging
import os
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Any
import matplotlib.pyplot as plt
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)

class PipelineStack:
    """Base class for pipeline stacks with parallel task execution."""
    def __init__(self, name, artifact_root, steps: List['Step'], max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = name
        self.steps = steps
        self.artifact_store: Dict[str, Artifact] = {}
        os.makedirs(artifact_root, exist_ok=True)

    def run_tasks(self, tasks: List[Callable]):
        """Execute tasks in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(task): task.__name__ for task in tasks}
            results = {}
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    results[task_name] = future.result()
                    self.logger.info(f"Task {task_name} completed successfully.")
                except Exception as e:
                    self.logger.error(f"Task {task_name} failed: {str(e)}")
            return results

    def save_pipeline(self, filename: str = "artifacts/pipeline_state.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        state = {
            "pipeline_name": self.name,
            "steps": [step.name for step in self.steps],
            "artifacts": {
                name: (
                    artifact.data.config_dict
                    if isinstance(artifact.data, Config)
                    else artifact.data
                )
                for name, artifact in self.artifact_store.items()
            },
        }
        with open(filename, "w") as file:
            json.dump(state, file, indent=4)
        self.logger.info(f"Pipeline state saved to {filename}")

    def visualize_pipeline(self, output_file: str = "artifacts/pipeline_graph"):
        G = nx.DiGraph()
        # Add nodes and edges for steps and artifacts
        for step in self.steps:
            G.add_node(step.name)
            for input_name in step.inputs:
                G.add_edge(input_name, step.name)
            for output_name in step.outputs:
                G.add_edge(step.name, output_name)

        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42)  # Layout algorithm for positioning
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=2000,
            font_size=10,
            font_weight="bold",
        )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(f"{output_file}.png")
        plt.show()
        self.logger.info(f"Pipeline visualization saved as {output_file}.png")

class Config:
    def __init__(self, config_dict: Dict = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None):
        return self.config_dict.get(key, default)

    @staticmethod
    def load_from_file(filename: str):
        """Loads configuration from a YAML or JSON file."""
        try:
            with open(filename, "r") as file:
                if filename.endswith((".yml", ".yaml")):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
            return Config(config_data)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {filename}: {e}")

class Artifact:
    """Placeholder class for artifacts."""
    def __init__(self, data):
        self.data = data

class Step:
    """Represents a step in the pipeline."""
    def __init__(self, name: str, inputs: List[str], outputs: List[str], task: Callable):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.task = task

class TrainingPipeline:
    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_from_file("config/config.yml")  # Ensure JSON loading

    def run(self):
        # Define steps
        data_ingestion_step = Step(
            name="data_ingestion",
            inputs=["path", "config"],
            outputs=["train_data", "test_data"],
            task=self.data_ingestion_task
        )
        data_validation_step = Step(
            name="data_validation",
            inputs=["train_data", "test_data", "config"],
            outputs=["val_train", "val_test"],
            task=self.data_validation_task
        )
        data_transformation_step = Step(
            name="data_transformation",
            inputs=["val_train", "val_test", "config"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            task=self.data_transformation_task
        )

        steps = [data_ingestion_step, data_validation_step, data_transformation_step]

        local_stack = PipelineStack(
            name="ml_customer_churn",
            artifact_root="artifacts",
            steps=steps,
            max_workers=2
        )

        # Run tasks
        tasks = [step.task for step in steps]
        local_stack.run_tasks(tasks)

        # Save and visualize pipeline
        local_stack.save_pipeline()
        local_stack.visualize_pipeline()

    def data_ingestion_task(self):
        self.logger.info("Running data ingestion task...")
        # Placeholder for actual data ingestion logic
        return {"train_data": "train_data", "test_data": "test_data"}

    def data_validation_task(self):
        self.logger.info("Running data validation task...")
        # Placeholder for actual data validation logic
        return {"val_train": "val_train", "val_test": "val_test"}

    def data_transformation_task(self):
        self.logger.info("Running data transformation task...")
        # Placeholder for actual data transformation logic
        return {"X_train": "X_train", "X_test": "X_test", "y_train": "y_train", "y_test": "y_test"}

if __name__ == "__main__":
    path = "data/churn-train.csv"
    pipe_instance = TrainingPipeline(path)
    pipe_instance.run()