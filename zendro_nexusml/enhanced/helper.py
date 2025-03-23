import time
import json
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Callable, List, Dict, Union
import uuid
import networkx as nx
from time import time

# Logger Setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Step to represent the configuration passed into the pipeline and execution.
class Config:
    def __init__(self, config_dict=None):
        self.config_dict = config_dict or {}

    def get(self, key, default=None):
        return self.config_dict.get(key, default)

    @staticmethod
    def load_from_file(filename):
        with open(filename, 'r') as file:
            config_data = json.load(file)
        return Config(config_data)


# Artifact class to handle outputs from each step and storage
class Artifact:
    def __init__(self, name, data=None):
        self.name = name
        self.data = data

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f)

    def load(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __repr__(self):
        return f"Artifact(name={self.name}, data={self.data})"


# Context class to manage the artifacts
class Context:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def update(self, key, artifact):
        self.data[key] = artifact
        logger.info(f"Artifact saved: {artifact.name}, data: {artifact.data}")



logger = logging.getLogger(__name__)

class Step:
    """Represents a step in the pipeline. Can also be used as a decorator."""

    # def __init__(self, function: Optional[Callable] = None, inputs: Optional[List[str]] = None,
                #  outputs: Optional[Union[str, List[str]]] = None, name: Optional[str] = None):
                 #,tags: Optional[List[str]] = None, description: Optional[str] = None): #config=None,
        # if function is not None:
        #     self.function = function
        #     self.name = name
            # self.config = config or Config()
            # self.inputs = inputs or []
            # self.outputs = [outputs] if isinstance(outputs, str) else (outputs or [])
            # self.tags = tags or []
            # self.executed = False
            # self.description = description or ''  # Set description to empty string if None
        # else:
        #     self.function = None  # Ensure consistent naming
        #     self.name = name
            # self.config = config or Config()
            # self.inputs = inputs or []
            # self.outputs = outputs or []
            # self.tags = tags or []
            # self.description = description or ''

    # def __call__(self, *args, **kwargs):
    #     """Allow the Step class to be used as a decorator."""
    #     def wrapper(func):
    #         return Step(function=func, inputs=self.inputs, outputs=self.outputs, config=self.config,
    #                     name=self.name, tags=self.tags, description=self.description)
    #     return wrapper

    # def __call__(self, *args, **kwargs):
    #     """Wrapper to call the function with the arguments."""
    #     return self.function(*args, **kwargs)
    def __init__(self, function, name, inputs=None, outputs=None):
        print(f"Initializing Step: {name}")
        if function is None:
            raise ValueError(f"Function for step '{name}' cannot be None")
        self.function = function
        self.name = name
        self.inputs = inputs or []
        self.outputs = outputs or []
        print(f"Function assigned: {self.function}")

    def __call__(self, *args, **kwargs):
        """Wrapper to call the function with the arguments."""
        if self.function is None:
            raise ValueError(f"Cannot call a None function for step '{self.name}'")
        return self.function(*args, **kwargs)
           

    def execute(self, input_artifacts: Dict, context=None):#, config: Optional[Dict] = None):
        """Execute the step's task."""
        if self.function:
            start_time = datetime.now()
            print(f"Executing step: {self.name}")

            # Execute the function assigned to this step
            result = self.function(*input_artifacts.values(), context=context)#, **config)#**(config or {}))

            duration = datetime.now() - start_time
            logger.info(f"Step {self.name} completed in {duration.total_seconds():.2f}s")

            self.executed = True
            print(f"Step {self.name} execution complete.")
            return result
        else:
            print(f"No function assigned to step: {self.name}")
            return None



class Pipeline:
    def __init__(self, name: str, description: str, max_workers: Optional[int] = None):
        """
        Initialize the pipeline.
        Args:
            name (str): Name of the pipeline.
            description (str): Description of the pipeline.
            max_workers (int, optional): Maximum number of workers for parallel execution.
        """
        self.name = name
        self.description = description
        self.steps: List[Step] = []  # List of nodes in the pipeline
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) if max_workers else None
        self.dag = nx.DiGraph()  # Directed Acyclic Graph for step dependencies
        self.id = str(uuid.uuid4())

    def add_step(self, step: Step, dependencies: Optional[List[Step]] = None):
        """Add a step to the pipeline, along with its dependencies."""
        self.steps.append(step)
        self.dag.add_node(step.name)

        # Add dependencies to the DAG
        if dependencies:
            for dep in dependencies:
                self.dag.add_edge(dep.name, step.name)

        # Verify that the DAG remains acyclic
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("Pipeline contains a cycle in the step dependencies.")

    def run(self, input_artifacts: Dict,  context: Optional[Context] = None):#config: Optional[Dict] = None,):
        """Run the entire pipeline."""
        logger.info(f"Running pipeline '{self.name}'...")

        # Use topological sorting to respect dependencies
        sorted_steps = list(nx.topological_sort(self.dag))

        try:
            if self.executor:
                self._run_in_parallel(sorted_steps, input_artifacts, context) #for config, 
            else:
                self._run_sequentially(sorted_steps, input_artifacts, context) #config, 
        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise

    def _run_sequentially(self, sorted_steps: List[str], input_artifacts: Dict, context: Optional[Context] = None): #config: Optional[Dict] = None, ):
        """Run steps one after another (sequential execution)."""
        for step_name in sorted_steps:
            step = next(s for s in self.steps if s.name == step_name)
            logger.info(f"Executing step '{step.name}' sequentially...")
            self._execute_step_with_dependencies(step, input_artifacts)#, config)
            step.execute(input_artifacts)#, config)

    def _run_in_parallel(self, sorted_steps: List[str], input_artifacts: Dict, context: Optional[Context] = None): #config: Optional[Dict] = None, ):
        """Run steps concurrently using ThreadPoolExecutor (parallel execution)."""
        futures = []

        for step_name in sorted_steps:
            step = next(s for s in self.steps if s.name == step_name)
            logger.info(f"Executing step '{step.name}' in parallel...")
            self._execute_step_with_dependencies(step, input_artifacts)#, config)

            # Submit step execution for parallel execution
            future = self.executor.submit(step.execute, input_artifacts)#, config)
            futures.append(future)

        # Wait for all tasks to complete
        for future in futures:
            future.result()  # This will block until the task is completed

    def _execute_step_with_dependencies(self, step: Step, input_artifacts: Dict):#, config: Optional[Dict] = None):
        """Ensure dependencies for the current step are handled before execution."""
        for dep_name in self.dag.predecessors(step.name):
            dep_step = next(s for s in self.steps if s.name == dep_name)
            if not dep_step.executed:
                logger.info(f"Executing dependency '{dep_step.name}' for step '{step.name}'...")
                dep_step.execute(input_artifacts)#, config)

        logger.info(f"All dependencies satisfied for step '{step.name}'.")





class DataIngestion:
    # Data Loading Step
    @Step(name='DataLoading', description='Load raw data from source', inputs=['source'], outputs=['train','test'])
    def load_data(source, config):
        """Load raw data from source."""
        # Simulate loading raw data
        return train, test

# class DataTransformation:
#     # Data Transformation Step
#     @Step(name='DataTransformer', description='Transform raw data into usable format', inputs=['train_data', 'test_data'], outputs=['transform_train', 'transform_test'])
#     def transform_data(train_data, test_data, config):
#         """Logic to transform train and test data."""
#         transform_train = f"transformed {train_data}"
#         transform_test = f"transformed {test_data}"
#         return transform_train, transform_test



# Flowpipe Class to manage the pipeline
# class Flowpipe:
#     def __init__(self, path: str, config: Config):
#         self.path = path
#         self.config = config
#         self.pipeline = Pipeline(name="preprocessing_pipeline", description="Pipeline for data preprocessing", max_workers=2)

#     def flow(self):
#         """Add steps to the pipeline."""
#         dataingestion = DataIngestion()
#         datatransformation = DataTransformation()

        
#         self.pipeline.add_step(dataingestion.load_data)
#         self.pipeline.add_step(datatransformation.transform_data, dependencies=[dataingestion])

    
#     def run(self):
#         """Run the pipeline with context."""
#         context = Context()
#         input_artifacts = {'source': self.path}
#         self.pipeline.run(input_artifacts, self.config.config_dict, context)
    

# # Main execution
# if __name__ == "__main__":
#     config = Config({'key': 'value'})  # Initialize config (you can load it from a file if needed)
#     path = 'data/data.csv'
#     flowpipe_instance = Flowpipe(path, config)
#     flowpipe_instance.flow()

#     flowpipe_instance.run()


