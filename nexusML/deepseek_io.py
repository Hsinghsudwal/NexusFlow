import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable

class PipelineStack:
    """Base class for pipeline stacks with parallel task execution."""
    def __init__(self, name: str, artifact_root: str, max_workers: int = 4):
        self.max_workers = max_workers
        self.name = name
        os.makedirs(artifact_root, exist_ok=True)
        self.artifact_store: Dict[str, Artifact] = {}


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
        logging.info(f"Pipeline state saved to {filename}")

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
        logging.info(f"Pipeline visualization saved as {output_file}.png")

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


    

    
    


from src.core.oi import Step, Pipeline, Stack, Config, Artifact
from src.ml_customer_churn.data_ingestion import DataIngestion
from src.ml_customer_churn.data_validation import DataValidation
from src.ml_customer_churn.data_transformation import DataTransforation

class TrainingPipeline:
    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_from_file("config/config.yml")  # Ensure JSON loading

    def run(self):
        local_stack = PipelineStack("ml_customer_churn", "artifacts", max_workers: bool)
        # local_stack = Stack("local", "artifacts/local")

        data_ingestion_1={
            train_data, test_data = DataIngestion().data_ingestion(path, config)}
        
        data_validation_2={
        val_train, val_test = DataValidation().data_validation(train_data, test_data, config)}

        data_transformer_3={
        X_train,X_test,y_train,y_test = DataTransforation().data_transformer(val_train, val_test,config)}

        pipeline = local_stack(
            [data_ingestion_1, data_validation_2, data_transformer_3],
        )


        

        pipeline.save_pipeline()
        pipeline.visualize_pipeline()

        # return pipeline


if __name__ == "__main__":
    path = "data/churn-train.csv"
    pipe_instance = TrainingPipeline(path)
    pipe_instance.run()






import logging
import os
import json
import yaml
import matplotlib.pyplot as plt
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Any

class Artifact:
    """Class to represent pipeline artifacts."""
    def __init__(self, name: str, data: Any):
        self.name = name
        self.data = data

class Step:
    """Class to represent a pipeline step."""
    def __init__(self, name: str, inputs: List[str], outputs: List[str], callable_func: Callable):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.callable_func = callable_func

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

class PipelineStack:
    """Base class for pipeline stacks with parallel task execution."""
    def __init__(self, name: str, artifact_root: str, max_workers: int = 4):
        self.max_workers = max_workers
        self.name = name
        self.artifact_root = artifact_root
        self.steps = []
        os.makedirs(artifact_root, exist_ok=True)
        self.artifact_store = {}
        self.logger = logging.getLogger(name)

    def __call__(self, steps_list: List[Dict]):
        """Create pipeline from step dictionaries."""
        # Convert dictionaries to Step objects
        for i, step_dict in enumerate(steps_list):
            for step_name, callable_func in step_dict.items():
                # For simplicity, using basic input/output identification
                # In a real implementation, you'd need more sophisticated input/output tracking
                inputs = []
                outputs = [f"output_{i}"]
                step = Step(step_name, inputs, outputs, callable_func)
                self.steps.append(step)
        return self

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

    def save_pipeline(self, filename: str = None):
        if filename is None:
            filename = os.path.join(self.artifact_root, "pipeline_state.json")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        state = {
            "pipeline_name": self.name,
            "steps": [step.name for step in self.steps],
            "artifacts": {
                name: (
                    artifact.data.config_dict
                    if isinstance(artifact.data, Config)
                    else str(artifact.data)  # Convert to string for JSON serialization
                )
                for name, artifact in self.artifact_store.items()
            },
        }
        with open(filename, "w") as file:
            json.dump(state, file, indent=4)
        logging.info(f"Pipeline state saved to {filename}")

    def visualize_pipeline(self, output_file: str = None):
        if output_file is None:
            output_file = os.path.join(self.artifact_root, "pipeline_graph")
            
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
        plt.close()  # Close the figure to avoid displaying it
        logging.info(f"Pipeline visualization saved as {output_file}.png")

# Example classes for ML pipeline
class DataIngestion:
    def data_ingestion(self, path, config):
        # Implementation details
        return "train_data", "test_data"

class DataValidation:
    def data_validation(self, train_data, test_data, config):
        # Implementation details
        return "val_train", "val_test"

class DataTransformation:  # Fixed typo in class name
    def data_transformer(self, val_train, val_test, config):
        # Implementation details
        return "X_train", "X_test", "y_train", "y_test"

class TrainingPipeline:
    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_from_file("config/config.yml")

    def run(self):
        # Fixed the syntax error with max_workers parameter
        local_stack = PipelineStack("ml_customer_churn", "artifacts", max_workers=4)
        
        # Fixed dictionary syntax
        data_ingestion_1 = {
            "data_ingestion": lambda: DataIngestion().data_ingestion(self.path, self.config)
        }
        
        data_validation_2 = {
            "data_validation": lambda: DataValidation().data_validation("train_data", "test_data", self.config)
        }
        
        data_transformer_3 = {
            "data_transformation": lambda: DataTransformation().data_transformer("val_train", "val_test", self.config)
        }

        pipeline = local_stack(
            [data_ingestion_1, data_validation_2, data_transformer_3]
        )

        pipeline.save_pipeline()
        pipeline.visualize_pipeline()

        return pipeline

    def run_tasks(self):
        # Implementation for the run_tasks method
        pipeline = self.run()
        # Execute the pipeline tasks
        tasks = [step.callable_func for step in pipeline.steps]
        return pipeline.run_tasks(tasks)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    path = "data/churn-train.csv"
    pipe_instance = TrainingPipeline(path)
    pipe_instance.run_tasks()
