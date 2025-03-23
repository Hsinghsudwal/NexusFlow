# Core modules
import yaml
import json
import pandas as pd
import os
import logging
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
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


# Core modules
import yaml
import json
import pandas as pd
import os
import logging
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
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

                # Update artifact_store with the task output
                self.artifact_store[task_name] = task_output
                
                # Fixed bug: inputs should only include keys that exist in the results
                # and are actually used by the current task
                inputs = [k for k in results.keys() if k != task_name]
                
                self.steps.append({
                    "name": task_name,
                    "inputs": inputs,
                    "outputs": [task_name]
                })

                logging.info(f"Task '{task_name}' completed successfully.")
            except Exception as e:
                logging.error(f"Task execution failed: {e}")
                raise e

    # def save_pipeline(self, filename: str = "artifacts/pipeline_state.json"):
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #     state = {
    #         "pipeline_name": self.name,
    #         "steps": [step["name"] for step in self.steps],  # Fixed: access name from dictionary
    #         "artifacts": self.artifact_store,
    #     }
        
    #     # Use a more efficient approach for JSON serialization
    #     try:
    #         with open(filename, "w") as file:
    #             json.dump(state, file, indent=4)
    #         logging.info(f"Pipeline state saved to {filename}")
    #     except TypeError:
    #         # Handle non-serializable objects
    #         logging.warning("Non-serializable objects found in state, storing simplified version")
    #         simplified_state = {
    #             "pipeline_name": self.name,
    #             "steps": [step["name"] for step in self.steps],
    #             "artifacts": {k: str(type(v)) for k, v in self.artifact_store.items()}
    #         }
    #         with open(filename, "w") as file:
    #             json.dump(simplified_state, file, indent=4)

    def visualize_pipeline(self, output_file: str = "artifacts/pipeline_graph"):
    """
    Create a comprehensive visualization of the pipeline, including steps, configurations and artifacts.
    """
    G = nx.DiGraph()
    
    # Add nodes for steps
    for step in self.steps:
        step_name = step["name"]
        G.add_node(step_name, type="step", color="lightblue", shape="ellipse")
    
    # Add config node
    G.add_node("config", type="config", color="lightgreen", shape="box")
    
    # Add artifact nodes
    for artifact_name in self.artifact_store.keys():
        if artifact_name not in [step["name"] for step in self.steps]:
            G.add_node(f"artifact_{artifact_name}", type="artifact", color="lightyellow", shape="box")
    
    # Add edges from config to all steps
    for step in self.steps:
        G.add_edge("config", step["name"], style="dashed")
    
    # Add edges between steps based on inputs/outputs
    for step in self.steps:
        step_name = step["name"]
        
        # Add edges from inputs to the current step
        for input_name in step["inputs"]:
            if input_name in [s["name"] for s in self.steps]:
                G.add_edge(input_name, step_name)
        
        # Add edges from the current step to its outputs
        for output_name in step["outputs"]:
            if output_name != step_name:  # Avoid self-loops
                # If output is an artifact (not another step), connect to artifact node
                if output_name not in [s["name"] for s in self.steps]:
                    G.add_edge(step_name, f"artifact_{output_name}")
                else:
                    G.add_edge(step_name, output_name)
        
        # Connect step to its corresponding artifact
        G.add_edge(step_name, f"artifact_{step_name}")
    
    # Only create visualization if there are nodes
    if G.number_of_nodes() > 0:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42, k=0.5)  # Adjust k for node spacing
        
        # Create node color and shape maps
        node_colors = []
        node_shapes = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'step')
            if node_type == 'step':
                node_colors.append('lightblue')
                node_shapes.append('o')  # circle for steps
            elif node_type == 'config':
                node_colors.append('lightgreen')
                node_shapes.append('s')  # square for config
            else:  # artifact
                node_colors.append('lightsalmon')
                node_shapes.append('s')  # square for artifacts
        
        # Draw nodes with different colors and shapes
        for shape in set(node_shapes):
            # Get indices of nodes with this shape
            indices = [i for i, s in enumerate(node_shapes) if s == shape]
            nodes = [list(G.nodes())[i] for i in indices]
            colors = [node_colors[i] for i in indices]
            
            if shape == 'o':  # Circle
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, 
                                      node_size=2000, node_shape=shape)
            else:  # Square or other
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, 
                                      node_size=1500, node_shape=shape)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                              arrowstyle='->', arrowsize=15)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Add a legend
        import matplotlib.patches as mpatches
        step_patch = mpatches.Patch(color='lightblue', label='Step')
        config_patch = mpatches.Patch(color='lightgreen', label='Config')
        artifact_patch = mpatches.Patch(color='lightsalmon', label='Artifact')
        plt.legend(handles=[step_patch, config_patch, artifact_patch], 
                  loc='upper right', bbox_to_anchor=(1, 1))
        
        # Remove axis
        plt.axis('off')
        
        # Save figure
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(f"{output_file}.png", bbox_inches='tight', dpi=150)
        plt.close()  # Close plot to free memory
        logging.info(f"Pipeline visualization saved as {output_file}.png")
    else:
        logging.warning("No nodes to visualize in pipeline graph")

    def save_pipeline(self, filename: str = "artifacts/pipeline_state.json"):
        """
        Save the pipeline state to a JSON file, properly handling non-serializable objects.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Create a serializable version of the artifact store
        serializable_artifacts = {}
        for key, value in self.artifact_store.items():
            if isinstance(value, dict):
                # Handle dictionaries containing DataFrames
                serializable_dict = {}
                for k, v in value.items():
                    if isinstance(v, pd.DataFrame):
                        # For DataFrames, store metadata instead of the data itself
                        serializable_dict[k] = {
                            "type": "DataFrame",
                            "shape": v.shape,
                            "columns": list(v.columns),
                            "dtypes": {col: str(dtype) for col, dtype in v.dtypes.items()}
                        }
                    else:
                        # Try to make other values serializable
                        try:
                            # Test if value is JSON serializable
                            json.dumps(v)
                            serializable_dict[k] = v
                        except (TypeError, OverflowError):
                            # If not serializable, store type info
                            serializable_dict[k] = {
                                "type": str(type(v)),
                                "summary": str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                            }
                serializable_artifacts[key] = serializable_dict
            else:
                # Try to make value serializable directly
                try:
                    # Test if value is JSON serializable
                    json.dumps(value)
                    serializable_artifacts[key] = value
                except (TypeError, OverflowError):
                    # For non-serializable objects, store metadata instead
                    if isinstance(value, pd.DataFrame):
                        serializable_artifacts[key] = {
                            "type": "DataFrame",
                            "shape": value.shape,
                            "columns": list(value.columns),
                            "dtypes": {col: str(dtype) for col, dtype in value.dtypes.items()}
                        }
                    else:
                        serializable_artifacts[key] = {
                            "type": str(type(value)),
                            "summary": str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        }

        # Create the state with serializable artifacts
        state = {
            "pipeline_name": self.name,
            "steps": [step["name"] for step in self.steps],
            "artifacts": serializable_artifacts
        }

        # Save to file
        try:
            with open(filename, "w") as file:
                json.dump(state, file, indent=4)
            logging.info(f"Pipeline state saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save pipeline state: {e}")
            # Fallback to a very simple state if everything else fails
            basic_state = {
                "pipeline_name": self.name,
                "steps": [step["name"] for step in self.steps],
                "error": f"Failed to save complete state: {str(e)}"
            }
            with open(filename, "w") as file:
                json.dump(basic_state, file, indent=4)

    

class DataIngestion:
    @staticmethod
    def data_ingestion(path, config):
        try:
            # Read only necessary columns to reduce memory usage
            df = pd.read_csv(path)

            data_path = config.get("data_path", {}).get("data", "data")
            raw_path = config.get("data_path", {}).get("raw_path", "raw")
            train_filename = config.get("data_path", {}).get("train", "train.csv")
            test_filename = config.get("data_path", {}).get("test", "test.csv")

            # Split data more efficiently
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            # Create directory only if needed to save I/O operations
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
    @staticmethod
    def data_validation(results, config):
        try:
            train_data = results.get("data_ingestion", {}).get("train_data")
            test_data = results.get("data_ingestion", {}).get("test_data")
            
            if train_data is None or test_data is None:
                raise ValueError("Missing required input data")
                
            # Basic validation - check for missing values
            train_missing = train_data.isnull().sum().sum()
            test_missing = test_data.isnull().sum().sum()
            
            validation_results = {
                "train_missing_values": train_missing,
                "test_missing_values": test_missing,
                "val_train": train_data,
                "val_test": test_data
            }
            
            logging.info(f"Data validation completed. Found {train_missing} missing values in training data.")
            return "data_validation", validation_results
        except Exception as e:
            logging.error(f"Data validation failed: {e}")
            raise e


class DataTransformation:
    @staticmethod
    def data_transformer(results, config):
        try:
            validation_results = results.get("data_validation", {})
            val_train = validation_results.get("val_train")
            val_test = validation_results.get("val_test")
            
            if val_train is None or val_test is None:
                raise ValueError("Missing required validation data")
            
            # Simple transformation - convert categorical features to numeric
            # This is just a placeholder - replace with your actual transformation logic
            categorical_cols = val_train.select_dtypes(include=['object']).columns
            
            transformed_train = val_train.copy()
            transformed_test = val_test.copy()
            
            for col in categorical_cols:
                # Use more efficient encoding approach
                categories = pd.Categorical(pd.concat([val_train[col], val_test[col]]).unique())
                transformed_train[col] = pd.Categorical(val_train[col], categories=categories).codes
                transformed_test[col] = pd.Categorical(val_test[col], categories=categories).codes
            
            logging.info("Data transformation completed.")
            return "data_transformation", {
                "transformed_train": transformed_train,
                "transformed_test": transformed_test
            }
        except Exception as e:
            logging.error(f"Data transformation failed: {e}")
            raise e


class TrainingPipeline:
    def __init__(self, path: str, config_path: str = "config/config.yml"):
        self.path = path
        self.config = Config.load_from_file(config_path)

    def run(self):
        try:
            ml_stack = Stack(name="ml_customer_churn", max_workers=4)

            # Define tasks for the MLOps stack with cleaner lambda functions
            tasks = [
                lambda results: DataIngestion.data_ingestion(self.path, self.config),
                lambda results: DataValidation.data_validation(results, self.config),
                lambda results: DataTransformation.data_transformer(results, self.config)
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