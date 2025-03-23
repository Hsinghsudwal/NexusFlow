from typing import Any, Dict
import json
import yaml
import datetime
import hashlib


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
                if filename.endswith(('.yml', '.yaml')):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
            return Config(config_data)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {filename}: {e}")


class Artifact:
    def __init__(self, data: Any, name: str):
        self.data = data
        self.name = name
        self.timestamp = datetime.datetime.now()
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Computes a hash for the stored data."""
        return hashlib.sha256(str(self.data).encode()).hexdigest()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns metadata containing timestamp and hash."""
        return {"timestamp": str(self.timestamp), "hash": self.hash}

    def __repr__(self):
        return f"Artifact(name={self.name}, hash={self.hash})"



import pandas as pd
import os
# from utils.constants import DATA, RAW_PATH, TRAIN, TEST
from sklearn.model_selection import train_test_split
from utils.helper import Config, Artifact
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# import prefect
# from prefect import task, Flow


class DataIngestion:

    def __init__(self) -> None:
        pass

    def data_ingestion(self, path, config):

        try:
            # os.makedirs("pipeline_data", exist_ok=True)

            # if isinstance(path, Artifact):
            #     path = path.data  # Ensure it's a string


            # if isinstance(config, Artifact):
            #     config = config.data


            df = pd.read_csv(path)
            
            data_path = config.get("data_path", {}).get("data")  
            raw_path = config.get("data_path", {}).get("raw_path")
            train_filename = config.get("data_path", {}).get("train")
            test_filename = config.get("data_path", {}).get("test")

            # print(f"data_path: {data_path}, raw_path: {raw_path}, train_filename: {train_filename}, test_filename: {test_filename}")


            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            logging.info("Data ingestion complete.")
            raw_data_path = os.path.join(data_path, raw_path)
            os.makedirs('local/',raw_data_path, exist_ok=True)

            # Save train and test data to CSV files
            train_data.to_csv(os.path.join(raw_data_path, train_filename), index=False)
            test_data.to_csv(os.path.join(raw_data_path, test_filename), index=False)

            logging.info("Data ingestion complete.")
            return train_data, test_data

        except Exception as e:
            raise e



import logging
from typing import Any, Callable, Dict, List
import concurrent.futures
import datetime
import hashlib
import json
import os
import time
import networkx as nx
import matplotlib.pyplot as plt
from utils.helper import Artifact, Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




class Step:
    def __init__(self, func, inputs=None, outputs=None, config=None):
        self.func = func
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.name = func.__name__
        self.config = config or {}

    def execute(self, artifacts):
        input_values = [artifacts[name].data for name in self.inputs]
        start_time = time.time()
        output_values = self.func(*input_values)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Step {self.name} executed in {execution_time:.2f} seconds.")
        if not isinstance(output_values, tuple):
            output_values = (output_values,)
        if len(output_values) != len(self.outputs):
            raise ValueError("Number of outputs does not match expected number")
        return {name: Artifact(data, name) for name, data in zip(self.outputs, output_values)}

class Pipeline:
    def __init__(self, name, steps):
        self.name = name
        self.steps = steps
        self.artifact_store = {}

    def run(self, stack, parallel=False):
        if parallel:
            self._run_parallel(stack)
        else:
            self._run_sequential(stack)

    def _run_parallel(self, stack):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for step in self.steps:
                if all(input_name in self.artifact_store for input_name in step.inputs):
                    futures[step] = executor.submit(step.execute, self.artifact_store)
                else:
                    logging.warning(f"Step {step.name} skipped due to unmet dependencies.")
            for future in concurrent.futures.as_completed(futures.values()):
                try:
                    results = future.result()
                    self.artifact_store.update(results)
                    stack.log_artifacts(results)
                except Exception as e:


    def _run_sequential(self, stack: "Stack"):
        for step in self.steps:
            if all(input_name in self.artifact_store for input_name in step.inputs):
                try:
                    results = step.execute(self.artifact_store)
                    self.artifact_store.update(results)
                    stack.log_artifacts(results)
                except Exception as e:
                    logging.error(f"Error in step execution: {e}", exc_info=True)
                    break  # Stop if error in sequential mode.
            else:
                logging.warning(f"Step {step.name} skipped due to unmet dependencies.")

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


class Stack:
    def __init__(self, name: str, artifact_root: str):
        self.name = name
        self.artifact_root = artifact_root
        os.makedirs(artifact_root, exist_ok=True)

    def log_artifacts(self, artifacts: Dict[str, Artifact]):
        for name, artifact in artifacts.items():
            artifact_path = os.path.join(self.artifact_root, f"{name}_{artifact.hash}.json")
            with open(artifact_path, "w") as f:
                json.dump({"data": str(artifact.data), "timestamp": str(artifact.timestamp)}, f)

    def deploy_model(self, model: Artifact, deployment_path: str):
        os.makedirs(deployment_path, exist_ok=True)
        model_path = os.path.join(deployment_path, f"{model.name}_{model.hash}.json")
        with open(model_path, "w") as f:
            json.dump({"data": str(model.data), "timestamp": str(model.timestamp)}, f)
        logging.info(f"Model deployed to {model_path}")

    def monitor_model(self, model: Artifact, data: Any) -> Any:
        if callable(model.data):
            return model.data(data)
        else:
            logging.error("Model data is not callable")
            return None



from utils.helper import Config, Artifact
from src.core.oi import Step, Pipeline, Stack

from src.ml_customer_churn.data_ingestion import DataIngestion
from src.ml_customer_churn.data_validation import DataValidation


# Training
class TrainingPipeline:
    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_from_file("config/config.yml")  # Ensure JSON loading

    def run(self):
        local_stack = Stack("ml_customer_churn", "artifacts/local")

        # Create Artifacts for path and config
        path_artifact = Artifact(self.path, "path")
        config_artifact = Artifact(self.config, "config")

        data_ingestion_step = Step(
            func=DataIngestion().data_ingestion,
            inputs=["path", "config"],
            outputs=["train_data", "test_data"],
            config=self.config
        )

        data_validation_step = Step(
            func=DataValidation().data_validation,
            inputs=["train_data", "test_data","config"],
            outputs=["val_train", "val_test"],
            config=self.config
        )

        pipeline = Pipeline("churn_pipeline", [data_ingestion_step, data_validation_step])

        # Add path and config artifacts to the artifact_store
        pipeline.artifact_store["path"] = path_artifact
        pipeline.artifact_store["config"] = config_artifact

        pipeline.run(local_stack, parallel=False)

        pipeline.save_pipeline()
        pipeline.visualize_pipeline()


if __name__ == "__main__":
    path = "data/churn-train.csv"
    pipe_instance = TrainingPipeline(path)
    pipe_instance.run()
