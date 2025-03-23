# Core modules
import yaml
import json
import pandas as pd
import os
import logging
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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


class Stack:
    def __init__(self, name: str, max_workers: int = 2):
        self.name = name
        self.max_workers = max_workers
        self.steps = []  # Initialize steps
        self.artifact_store = {}  # Initialize artifact store

    def run_tasks(self, tasks: List[callable]):
        """Executes tasks sequentially, passing outputs from one task to the next."""
        results = {}
        for task in tasks:
            try:
                # Execute the task and store its results
                task_name, task_output = task(results)
                results[task_name] = task_output
                logging.info(f"Task '{task_name}' completed successfully.")
            except Exception as e:
                logging.error(f"Task execution failed: {e}")
                raise e

    def save_pipeline(self, filename: str = "artifacts/pipeline_state.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        state = {
            "pipeline_name": self.name,
            "steps": [step.name for step in self.steps],
            "artifacts": {
                name: (
                    artifact.data.config_dict if isinstance(artifact.data, Config) else artifact.data
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

        
class DataIngestion:
    def __init__(self) -> None:
        pass

    def data_ingestion(self, path, config):
        try:
            df = pd.read_csv(path)

            data_path = config.get("data_path", {}).get("data")
            raw_path = config.get("data_path", {}).get("raw_path")
            train_filename = config.get("data_path", {}).get("train")
            test_filename = config.get("data_path", {}).get("test")

            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            raw_data_path = os.path.join(data_path, raw_path)
            os.makedirs(raw_data_path, exist_ok=True)

            # Save train and test data to CSV files
            train_data.to_csv(os.path.join(raw_data_path, train_filename), index=False)
            test_data.to_csv(os.path.join(raw_data_path, test_filename), index=False)

            logging.info("Data ingestion completed.")
            return "data_ingestion", {"train_data": train_data, "test_data": test_data}
        except Exception as e:
            logging.error(f"Data ingestion failed: {e}")
            raise e


class DataValidation:
    def __init__(self) -> None:
        pass

    def data_validation(self, results, config):
        try:
            train_data = results["data_ingestion"]["train_data"]
            test_data = results["data_ingestion"]["test_data"]
            
            # Implement your validation logic here
            # For example: check for missing values, data types, etc.
            
            logging.info("Data validation completed.")
            return "data_validation", {"val_train": train_data, "val_test": test_data}
        except Exception as e:
            logging.error(f"Data validation failed: {e}")
            raise e


class DataTransformation:
    def __init__(self) -> None:
        pass

    def data_transformer(self, results, config):
        try:
            val_train = results["data_validation"]["val_train"]
            val_test = results["data_validation"]["val_test"]
            
            # Implement your transformation logic here
            # For example: feature engineering, normalization, etc.
            
            logging.info("Data transformation completed.")
            return "data_transformation", {"transformed_train": val_train, "transformed_test": val_test}
        except Exception as e:
            logging.error(f"Data transformation failed: {e}")
            raise e


class TrainingPipeline:
    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_from_file("config/config.yml")

    def run(self):
        try:
            ml_stack = Stack(name="ml_customer_churn", max_workers=4)

            # Define tasks for the MLOps stack
            tasks = [
                lambda results: DataIngestion().data_ingestion(self.path, self.config),
                lambda results: DataValidation().data_validation(results, self.config),
                lambda results: DataTransformation().data_transformer(results, self.config)
            ]

            # Run tasks sequentially
            ml_stack.run_tasks(tasks)

            # Save and visualize the pipeline
            ml_stack.save_pipeline()
            ml_stack.visualize_pipeline()

            return True
        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            return False


if __name__ == "__main__":
    path = "data/churn-train.csv"
    pipeline = TrainingPipeline(path)
    if pipeline.run():
        print("Pipeline executed successfully!")
    else:
        print("Pipeline execution failed!")