import os
import json
import yaml
import logging
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from typing import List, Dict, Callable, Any, Optional


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
    """Class for storing pipeline artifacts."""
    def __init__(self, data, version: str = None):
        self.data = data
        self.version = version or datetime.now().strftime("%Y%m%d%H%M%S")
        self.created_at = datetime.now()

    # def get_artifact(self, name: str):
    #     return self.artifact_store.get(name, None)


class Step:
    """Represents a step in the pipeline."""
    def __init__(self, name: str, inputs: List[str], outputs: List[str], task: Callable, critical: bool = False):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.task = task
        self.critical = critical  # Flag to indicate if step failure should stop pipeline
        # Store the task name as the function name
        self.task.__name__ = name


class PipelineStack:
    """Base class for pipeline stacks with parallel task execution."""
    def __init__(self, name, artifact_root, steps: List['Step'], max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = name
        self.steps = steps
        self.artifact_store: Dict[str, Artifact] = {}
        self.artifact_root = artifact_root
        os.makedirs(artifact_root, exist_ok=True)
        # self.serializer = ModelSerializer(os.path.join(artifact_root, "models"))
        # self.drift_detector = None


    def run_tasks(self, tasks: List[Callable]):
        """Execute tasks in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(task): task.__name__ for task in tasks}
            results = {}
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    task_results = future.result()
                    if task_results and isinstance(task_results, dict):
                        # Store task results as artifacts
                        for name, data in task_results.items():
                            self.artifact_store[name] = Artifact(data)
                            self.logger.info(f"Artifact '{name}' created and stored")
                    results[task_name] = task_results
                    self.logger.info(f"Task {task_name} completed successfully.")
                except Exception as e:
                    self.logger.error(f"Task {task_name} failed: {str(e)}")
                    # Propagate exception to stop pipeline if critical
                    if task_name in [step.name for step in self.steps]:
                        step = next(step for step in self.steps if step.name == task_name)
                        if step.critical:
                            raise RuntimeError(f"Critical step {task_name} failed: {str(e)}")
            return results

    def save_pipeline(self, filename: Optional[str] = None):
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
                    else artifact.data
                )
                for name, artifact in self.artifact_store.items()
                if hasattr(artifact, 'data')  # Check if the artifact has data attribute
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
        
    def get_artifact(self, name: str):
        return self.artifact_store.get(name, None)


class DataIngestion:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def data_ingestion(self, path, config):
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
        self.logger.info("Data ingestion completed.")
        return {"train_data": train_data, "test_data": test_data}


# Placeholder classes for the imported modules
class DataValidation:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def data_validation(self, train_data, test_data, config):
        # Add your data validation logic here
        self.logger.info("Data validation completed.")
        return {"val_train": train_data, "val_test": test_data}


class DataTransformation:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def data_transformation(self, val_train, val_test, config):
        # Add your data transformation logic here
        # For example, separating features and target
        X_train = val_train.drop('Churn', axis=1) if 'Churn' in val_train.columns else val_train
        y_train = val_train['Churn'] if 'Churn' in val_train.columns else None
        X_test = val_test.drop('Churn', axis=1) if 'Churn' in val_test.columns else val_test
        y_test = val_test['Churn'] if 'Churn' in val_test.columns else None
        
        self.logger.info("Data transformation completed.")
        return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


class TrainingPipeline:
    def __init__(self, path: str):
        self.path = path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = Config.load_from_file("config/config.yml")
        

    def run(self):
        # Define steps
        data_ingestion_step = Step(
            name="data_ingestion",
            inputs=["path", "config"],
            outputs=["train_data", "test_data"],
            task=DataIngestion().data_ingestion,
            critical=True  # Mark as critical - pipeline will stop if this fails
        )
        data_validation_step = Step(
            name="data_validation",
            inputs=["train_data", "test_data", "config"],
            outputs=["val_train", "val_test"],
            task=DataValidation().data_validation
        )
        data_transformation_step = Step(
            name="data_transformation",
            inputs=["val_train", "val_test", "config"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            task=DataTransformation().data_transformation
        )

        steps = [data_ingestion_step, data_validation_step, data_transformation_step]

        # Create stack with steps
        self.pipeline_stack = PipelineStack(
            name="ml_customer_churn",
            artifact_root="artifacts",
            steps=steps,
            max_workers=2
        )
        
        # Add path to artifact store
        self.pipeline_stack.artifact_store["path"] = Artifact(self.path)
        self.pipeline_stack.artifact_store["config"] = Artifact(self.config)

        try:
            # Run tasks
            self.pipeline_stack.run_tasks([step.task for step in steps])

            # Save and visualize pipeline
            self.pipeline_stack.save_pipeline()
            self.pipeline_stack.visualize_pipeline()
            
            self.logger.info("Pipeline completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    path = "data/churn-train.csv"
    pipeline = TrainingPipeline(path)
    if pipeline.run():
        print("Pipeline executed successfully!")
        # Uncomment to check retraining status
        # retraining_status = pipeline.pipeline_stack.get_artifact("retraining_decision")
        # if retraining_status and retraining_status.data.get('required'):
        #     print("System requires retraining based on:")
        #     print(json.dumps(retraining_status.data, indent=2))
    else:
        print("Pipeline execution failed")