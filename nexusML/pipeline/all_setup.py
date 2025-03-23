import json
import logging
import os
import graphviz
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split

# Config Management
class Config:
    def __init__(self, config_dict: Dict = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None):
        return self.config_dict.get(key, default)

    @staticmethod
    def load_from_file(filename: str):
        with open(filename, 'r') as file:
            config_data = json.load(file)
        return Config(config_data)


# Artifact Management
class Artifact:
    def __init__(self, name: str, data: Any = None):
        self.name = name
        self.data = data

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.data, f)

    def load(self, path: str):
        with open(path, 'r') as f:
            self.data = json.load(f)

    def upload(self, store, path: str):
        store.save(self, path)

    def download(self, store, path: str):
        return store.load(path)

    def __repr__(self):
        return f"Artifact(name={self.name}, data={self.data})"


# Context to Handle Artifacts During Execution
class Context:
    def __init__(self):
        self.data = {}

    def get(self, key: str):
        return self.data.get(key)

    def update(self, key: str, artifact: Artifact):
        self.data[key] = artifact
        logging.info(f"Artifact saved: {artifact.name}, data: {artifact.data}")

def display_context_contents(context: Context):
    logging.info("Pipeline Execution Context Contents:")
    for key, artifact in context.data.items():
        logging.info(f"Artifact Name: {artifact.name}")
        logging.info(f"Data Type: {type(artifact.data)}")
        if isinstance(artifact.data, pd.DataFrame):  # If it's a DataFrame, show shape and head
            logging.info(f"Shape: {artifact.data.shape}")
            logging.info(f"Preview:\n{artifact.data.head()}")
        else:
            logging.info(f"Data: {artifact.data}")


# Pipeline Step Interface
class PipelineStep:
    def __init__(self, name: str, function, inputs=None, outputs=None):
        self.name = name
        self.function = function
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.execution_time = None

    def execute(self, context: Context):
        logging.info(f"Executing step: {self.name}")
        input_data = {key: context.get(key) for key in self.inputs}
        output_data = self.function(**input_data)

        if isinstance(output_data, tuple):
            for key, data in zip(self.outputs, output_data):
                context.update(key, Artifact(name=key, data=data))
        else:
            context.update(self.outputs[0], Artifact(name=self.outputs[0], data=output_data))

        return output_data


# Pipeline Execution
class Pipeline:
    def __init__(self, name: str, config: Config, max_workers: int = 2):
        self.name = name
        self.config = config
        self.steps = []
        self.max_workers = max_workers
        self.execution_metadata = {}

    def add_step(self, pipestep: PipelineStep):
        self.steps.append(pipestep)

    def execute(self, execution_context: Dict):
        context = Context()

        # Initialize execution context
        for key, value in execution_context.items():
            context.update(key, Artifact(name=key, data=value))

        logging.info(f"Executing pipeline '{self.name}' with {self.max_workers} workers.")

        # Parallel execution
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
        with open(filename, "w") as file:
            json.dump(self.execution_metadata, file, indent=4)
        logging.info(f"Pipeline state saved to {filename}")

    def visualize_pipeline(self, output_file: str = "pipeline_graph"):
        dot = graphviz.Digraph(comment=self.name)

        for step in self.steps:
            dot.node(step.name, step.name)
            for input_name in step.inputs:
                dot.edge(input_name, step.name)
            for output_name in step.outputs:
                dot.edge(step.name, output_name)

        dot.render(output_file, format="png", cleanup=True)
        logging.info(f"Pipeline visualization saved as {output_file}.png")


# Data Ingestion Class
class DataIngestion:
    def __init__(self):
        pass

    def data_ingestion(self, path: str, config: Config):
        try:
            df = pd.read_csv(path)

            data_path = config.get("data_location_path", {}).get("data")
            raw_path = config.get("data_location_path", {}).get("raw_path")
            train_filename = config.get("data_location_path", {}).get("train")
            test_filename = config.get("data_location_path", {}).get("test")

            if not all([data_path, raw_path, train_filename, test_filename]):
                raise ValueError("Missing required configuration values.")

            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            raw_data_path = os.path.join(data_path, raw_path)
            os.makedirs(raw_data_path, exist_ok=True)

            train_data.to_csv(os.path.join(raw_data_path, train_filename), index=False)
            test_data.to_csv(os.path.join(raw_data_path, test_filename), index=False)

            return train_data, test_data

        except Exception as e:
            logging.error(f"Data ingestion failed: {e}")
            raise e


# Training Pipeline
class TrainingPipeline:
    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_from_file("config/config.json")  # Ensure JSON loading

    def run(self):
        pipeline = Pipeline(name="ml_customer_churn", config=self.config, max_workers=5)
        ingestion_step = PipelineStep(
            name="data_ingestion",
            function=DataIngestion().data_ingestion,
            inputs=["path", "config"],
            outputs=["train", "test"]
        )
        pipeline.add_step(ingestion_step)

        execution_context = {"path": self.path, "config": self.config}
        context = pipeline.execute(execution_context)

        pipeline.save_pipeline()
        pipeline.visualize_pipeline()
            # access train and test data from the context.
            # train_data = context.get('train').data
            # test_data = context.get('test').data
            # print("train data shape", train_data.shape)
            # print("test data shape", test_data.shape)
        # return pipeline.execute({"path": self.path, "config": self.config})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = "data/churn-train.csv"
    pipe_instance = TrainingPipeline(path)
    pipe_instance.run()




from utils.helper import Config, Artifact, Context

import yaml
import json
import time
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Pipeline Step Interface
class PipelineStep:
    def __init__(self, name: str, function, inputs=None, outputs=None):
        self.name = name
        self.function = function
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.execution_time = None


    def execute(self, context: Context):
        logging.info(f"Executing step: {self.name}")
        start_time = time.time()

        # Get input data
        input_data = {key: context.get(key) for key in self.inputs}
        logging.info(f"{self.name} received inputs: {list(input_data.keys())}")

        # Execute function
        output_data = self.function(**input_data)

        logging.info(f"{self.name} produced output: {type(output_data)}")

        # Store output in context
        if isinstance(output_data, tuple):
            if len(output_data) != len(self.outputs):
                raise ValueError(f"{self.name} expected {len(self.outputs)} outputs, but got {len(output_data)}")

            for key, data in zip(self.outputs, output_data):
                logging.info(f"Storing output {key} in context.")
                context.update(key, Artifact(name=key, data=data))

        else:
            if len(self.outputs) != 1:
                raise ValueError(f"{self.name} expected 1 output, but got {len(self.outputs)}")

            logging.info(f"Storing single output {self.outputs[0]} in context.")
            context.update(self.outputs[0], Artifact(name=self.outputs[0], data=output_data))

        self.execution_time = time.time() - start_time
        return output_data



# Pipeline Execution
class Pipeline:
    def __init__(self, name: str, config: Config, max_workers: int = 2):
        self.name = name
        self.config = config
        self.steps = []
        self.max_workers = max_workers
        self.execution_metadata = {}

    def add_step(self, pipestep: PipelineStep):
        self.steps.append(pipestep)

    def execute(self, execution_context: Dict):
        context = Context()

        # Initialize execution context
        for key, value in execution_context.items():
            context.update(key, Artifact(name=key, data=value))

        logging.info(
            f"Executing pipeline '{self.name}' with {self.max_workers} workers."
        )

        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_step = {
                executor.submit(step.execute, context): step for step in self.steps
            }

            for future in as_completed(future_to_step):
                step = future_to_step[future]
                try:
                    result = future.result()

                    logging.info(f"Step {step.name} completed.")
                    logging.info(f"Context keys after {step.name}: {list(context.data.keys())}")
                    if "train_data" in context.data:
                        train_data = context.get("train_data")
                        logging.info(f"train_data exists, Type: {type(train_data)}, Shape: {train_data.data.shape if isinstance(train_data.data, pd.DataFrame) else 'Not a DataFrame'}")
                    else:
                        logging.error("train_data is missing in context!")

                    if "test_data" in context.data:
                        test_data = context.get("test_data")
                        logging.info(f"test_data exists, Type: {type(test_data)}, Shape: {test_data.data.shape if isinstance(test_data.data, pd.DataFrame) else 'Not a DataFrame'}")
                    else:
                        logging.error("test_data is missing in context!")


                    self.execution_metadata[step.name] = {
                        "execution_time": step.execution_time,
                        "inputs": step.inputs,
                        "outputs": step.outputs,
                    }
                except Exception as e:
                    logging.error(f"Step {step.name} failed: {e}")

        return context

    def save_pipeline(self, filename: str = "pipeline_data/pipeline_state.json"):
        with open(filename, "w") as file:
            json.dump(self.execution_metadata, file, indent=4)
        logging.info(f"Pipeline state saved to {filename}")

    def visualize_pipeline(self, output_file: str = "pipeline_data/pipeline_graph"):
        G = nx.DiGraph()

        # Add nodes and edges
        for step in self.steps:
            G.add_node(step.name)
            for input_name in step.inputs:
                G.add_edge(input_name, step.name)
            for output_name in step.outputs:
                G.add_edge(step.name, output_name)

        # Draw the graph
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

        # Save and show
        plt.savefig(f"{output_file}.png")
        plt.show()
        logging.info(f"Pipeline visualization saved as {output_file}.png")
