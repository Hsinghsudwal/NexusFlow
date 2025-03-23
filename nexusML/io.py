import logging
from typing import Any, Callable, Dict, List
import concurrent.futures
import datetime
import hashlib
import os
import time
import networkx as nx
import matplotlib.pyplot as plt
import json
import yaml


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


class Step:
    def __init__(self, func, inputs=None, outputs=None, config=None):
        self.func = func
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.name = func.__name__
        self.config = config or {}

    def execute(self, artifacts):
        print("Executing Step.execute...")
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
        return {
            name: Artifact(data, name)
            for name, data in zip(self.outputs, output_values)
        }


class Pipeline:
    def __init__(self, name: str, steps: List[Step], max_workers: int = 4):
        self.name = name
        self.steps = steps
        self.max_workers = max_workers
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


class Stack:
    def __init__(self, name: str, artifact_root: str):
        self.name = name
        self.artifact_root = artifact_root
        os.makedirs(artifact_root, exist_ok=True)

    # def log_artifacts(self, artifacts: Dict[str, Artifact]):
    #     for name, artifact in artifacts.items():
    #         artifact_path = os.path.join(
    #             self.artifact_root, f"{name}_{artifact.hash}.json"
    #         )
    #         with open(artifact_path, "w") as f:
    #             json.dump(
    #                 {"data": str(artifact.data), "timestamp": str(artifact.timestamp)},
    #                 f,
    #             )

    # def deploy_model(self, model: Artifact, deployment_path: str):
    #     os.makedirs(deployment_path, exist_ok=True)
    #     model_path = os.path.join(deployment_path, f"{model.name}_{model.hash}.json")
    #     with open(model_path, "w") as f:
    #         json.dump({"data": str(model.data), "timestamp": str(model.timestamp)}, f)
    #     logging.info(f"Model deployed to {model_path}")

    # def monitor_model(self, model: Artifact, data: Any) -> Any:
    #     # Simple monitoring example: run the model if it is callable
    #     if callable(model.data):
    #         return model.data(data)
    #     else:
    #         logging.error("Model data is not callable")
    #         return None



from src.core.oi import Step, Pipeline, Stack, Config, Artifact
from src.ml_customer_churn.data_ingestion import DataIngestion
from src.ml_customer_churn.data_validation import DataValidation
from src.ml_customer_churn.data_transformation import DataTransforation


# Training
class TrainingPipeline:
    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_from_file("config/config.yml")  # Ensure JSON loading

    def run(self):
        local_stack = Stack("ml_customer_churn", "artifacts")
        # local_stack = Stack("local", "artifacts/local")

        path_artifact = Artifact(self.path, "path")
        config_artifact = Artifact(self.config, "config")

        data_ingestion_step = Step(
            func=DataIngestion().data_ingestion,
            inputs=["path", "config"],
            outputs=["train_data", "test_data"],
            config=self.config,
        )

        data_validation_step = Step(
            func=DataValidation().data_validation,
            inputs=["train_data", "test_data", "config"],
            outputs=["val_train", "val_test"],
            config=self.config,
        )

        data_transformer_step = Step(
            func=DataTransforation().data_transformer,
            inputs=["val_train", "val_test", "config"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            config=self.config,
        )

        # Create pipeline
        pipeline = Pipeline(
            "churn_pipeline",
            [data_ingestion_step, data_validation_step, data_transformer_step],
        )

        pipeline.artifact_store["path"] = path_artifact
        pipeline.artifact_store["config"] = config_artifact

        pipeline.run_tasks(local_stack, max_workers=True)

        pipeline.save_pipeline()
        pipeline.visualize_pipeline()


if __name__ == "__main__":
    path = "data/churn-train.csv"
    pipe_instance = TrainingPipeline(path)
    pipe_instance.run()























# class CloudStack(Stack):
#     def log_artifacts(self, artifacts: Dict[str, Artifact]):
#         import boto3
#         s3 = boto3.client('s3')
#         bucket_name = "your-s3-bucket"
#         for name, artifact in artifacts.items():
#             object_key = f"{self.artifact_root}/{name}_{artifact.hash}.json"
#             s3.put_object(
#                 Bucket=bucket_name,
#                 Key=object_key,
#                 Body=json.dumps({"data": str(artifact.data), "timestamp": str(artifact.timestamp)}).encode()
#             )
#         logging.info("Artifacts logged to cloud storage.")

#     def deploy_model(self, model: Artifact, deployment_path: str):
#         logging.info(f"Model deployed to Lambda using path: {deployment_path}")

#     def monitor_model(self, model: Artifact, data: Any) -> Any:
#         logging.info(f"Monitoring prediction using CloudWatch for data: {data}")
#         return super().monitor_model(model, data)


# Example ML functions as standalone functions
# Alternatively, you can use classes with static methods:
# class LoadData:
#     @staticmethod
#     def load_data():
#         return [1, 2, 3, 4, 5]

# class TrainModel:
#     @staticmethod
#     def train_model(data):
#         def simple_model(x):
#             return x * 2
#         return simple_model

# class EvaluateModel:
#     @staticmethod
#     def evaluate_model(model, data):
#         results = [model(x) for x in data]
#         return sum(results) / len(results)


# Example Usage
# if __name__ == "__main__":
#     # Use local stack
#     local_stack = Stack("local", "artifacts/local")

#     # Define steps using the class-based static methods
#     load_step = Step(LoadData.load_data, outputs=["data"])
#     train_step = Step(TrainModel.train_model, inputs=["data"], outputs=["model"], config={"learning_rate": 0.2})
#     eval_step = Step(EvaluateModel.evaluate_model, inputs=["model", "data"], outputs=["evaluation"])

#     # Create pipeline
#     pipeline = Pipeline("example_pipeline", [load_step, train_step, eval_step])

#     # Run pipeline in parallel
#     logging.info("Running pipeline in parallel:")
#     start_time_parallel = time.time()
#     pipeline.run(local_stack, parallel=True)
#     end_time_parallel = time.time()
#     logging.info(f"Parallel execution time: {end_time_parallel - start_time_parallel:.2f} seconds")

#     # Reset artifact store for sequential run
#     pipeline.artifact_store = {}

# Run pipeline sequentially
# logging.info("Running pipeline sequentially:")
# start_time_sequential = time.time()
# pipeline.run(local_stack, parallel=False)
# end_time_sequential = time.time()
# logging.info(f"Sequential execution time: {end_time_sequential - start_time_sequential:.2f} seconds")

# # Deployment and Monitoring
# if "model" in pipeline.artifact_store:
#     trained_model = pipeline.artifact_store["model"]
#     local_stack.deploy_model(trained_model, "deployments/local")

#     test_data = 6
#     prediction = local_stack.monitor_model(trained_model, test_data)
#     logging.info(f"Prediction for {test_data}: {prediction}")
# else:
#     logging.error("Model artifact not found in pipeline artifact store.")
