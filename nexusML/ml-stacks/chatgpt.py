import concurrent.futures
import datetime
import hashlib
import json
import os
import time
import logging
from typing import Any, Callable, Dict, List

import networkx as nx
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Artifact:
    def __init__(self, data: Any, name: str):
        self.data = data
        self.name = name
        self.timestamp = datetime.datetime.now()
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        return hashlib.sha256(str(self.data).encode()).hexdigest()

    @property
    def metadata(self) -> Dict[str, Any]:
        return {"timestamp": str(self.timestamp), "hash": self.hash}

    def __repr__(self):
        return f"Artifact(name={self.name}, hash={self.hash})"


class Step:
    def __init__(self, func: Callable, inputs: List[str] = None, outputs: List[str] = None, config: Dict = None):
        self.func = func
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.name = func.__name__
        self.config = config or {}

    def execute(self, artifacts: Dict[str, Artifact]) -> Dict[str, Artifact]:
        # Gather input values from artifact store
        input_values = [artifacts[name].data for name in self.inputs]
        start_time = time.time()  # Start timing the function
        output_values = self.func(*input_values)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Step {self.name} executed in {execution_time:.2f} seconds.")

        # Ensure the outputs are in tuple format for consistent processing
        if not isinstance(output_values, tuple):
            output_values = (output_values,)

        if len(output_values) != len(self.outputs):
            raise ValueError("Number of outputs does not match expected number")

        return {name: Artifact(data, name) for name, data in zip(self.outputs, output_values)}


class Pipeline:
    def __init__(self, name: str, steps: List[Step]):
        self.name = name
        self.steps = steps
        self.artifact_store: Dict[str, Artifact] = {}

    def run(self, stack: "Stack", parallel: bool = True):
        if parallel:
            self._run_parallel(stack)
        else:
            self._run_sequential(stack)

    def _run_parallel(self, stack: "Stack"):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for step in self.steps:
                # Check if all dependencies are met
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
                    logging.error(f"Error in step execution: {e}", exc_info=True)

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
        # Save the artifact store metadata as the pipeline state
        with open(filename, "w") as file:
            json.dump({name: artifact.metadata for name, artifact in self.artifact_store.items()}, file, indent=4)
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
        # Simple monitoring example: run the model if it is callable
        if callable(model.data):
            return model.data(data)
        else:
            logging.error("Model data is not callable")
            return None


class CloudStack(Stack):
    def log_artifacts(self, artifacts: Dict[str, Artifact]):
        import boto3
        s3 = boto3.client('s3')
        bucket_name = "your-s3-bucket"
        for name, artifact in artifacts.items():
            object_key = f"{self.artifact_root}/{name}_{artifact.hash}.json"
            s3.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=json.dumps({"data": str(artifact.data), "timestamp": str(artifact.timestamp)}).encode()
            )
        logging.info("Artifacts logged to cloud storage.")

    def deploy_model(self, model: Artifact, deployment_path: str):
        logging.info(f"Model deployed to Lambda using path: {deployment_path}")

    def monitor_model(self, model: Artifact, data: Any) -> Any:
        logging.info(f"Monitoring prediction using CloudWatch for data: {data}")
        return super().monitor_model(model, data)


# Example ML functions as standalone functions

def load_data():
    return [1, 2, 3, 4, 5]

def train_model(data):
    def simple_model(x):
        return x * 2
    return simple_model

def evaluate_model(model, data):
    results = [model(x) for x in data]
    return sum(results) / len(results)


# Alternatively, you can use classes with static methods:
class LoadData:
    @staticmethod
    def load_data():
        return [1, 2, 3, 4, 5]

class TrainModel:
    @staticmethod
    def train_model(data):
        def simple_model(x):
            return x * 2
        return simple_model

class EvaluateModel:
    @staticmethod
    def evaluate_model(model, data):
        results = [model(x) for x in data]
        return sum(results) / len(results)


# Example Usage
if __name__ == "__main__":
    # Use local stack
    local_stack = Stack("local", "artifacts/local")

    # Define steps using the class-based static methods
    load_step = Step(LoadData.load_data, outputs=["data"])
    train_step = Step(TrainModel.train_model, inputs=["data"], outputs=["model"], config={"learning_rate": 0.2})
    eval_step = Step(EvaluateModel.evaluate_model, inputs=["model", "data"], outputs=["evaluation"])

    # Create pipeline
    pipeline = Pipeline("example_pipeline", [load_step, train_step, eval_step])

    # Run pipeline in parallel
    logging.info("Running pipeline in parallel:")
    start_time_parallel = time.time()
    pipeline.run(local_stack, parallel=True)
    end_time_parallel = time.time()
    logging.info(f"Parallel execution time: {end_time_parallel - start_time_parallel:.2f} seconds")

    # Reset artifact store for sequential run
    pipeline.artifact_store = {}

    # Run pipeline sequentially
    logging.info("Running pipeline sequentially:")
    start_time_sequential = time.time()
    pipeline.run(local_stack, parallel=False)
    end_time_sequential = time.time()
    logging.info(f"Sequential execution time: {end_time_sequential - start_time_sequential:.2f} seconds")

    # Deployment and Monitoring
    if "model" in pipeline.artifact_store:
        trained_model = pipeline.artifact_store["model"]
        local_stack.deploy_model(trained_model, "deployments/local")

        test_data = 6
        prediction = local_stack.monitor_model(trained_model, test_data)
        logging.info(f"Prediction for {test_data}: {prediction}")
    else:
        logging.error("Model artifact not found in pipeline artifact store.")
