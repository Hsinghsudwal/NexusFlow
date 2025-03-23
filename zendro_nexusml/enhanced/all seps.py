# mlops_framework/steps.py

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Optional

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


# Step class that includes configuration and artifact handling
class Step:
    def __init__(self, name, function, inputs=None, outputs=None, config=None):
        self.name = name
        self.function = function
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.config = config or Config()

    def __call__(self, *args, **kwargs):
        """Allow the Node class to be used as a decorator."""
        return Step(function=self.function, inputs=self.inputs, outputs=self.outputs, name=self.name, config=self.config)

    def execute(self, context):
        """Execute the step's task."""
        logger.info(f"Executing step: {self.name}")
        start_time = datetime.now()

        # Retrieve input artifacts from the context
        input_data = [context.get(input_name) for input_name in self.inputs]
        
        # Call the function with inputs and config
        try:
            result = self.function(*input_data, config=self.config)
        except Exception as e:
            logger.error(f"Error executing step {self.name}: {e}")
            raise
        
        # Create and store output artifacts
        for output_name in self.outputs:
            artifact = Artifact(output_name, result)
            context.update(output_name, artifact)

        # Log the duration of the step execution
        duration = datetime.now() - start_time
        logger.info(f"Step {self.name} completed in {duration.total_seconds():.2f}s")
        return result

    def execute_async(self, context):
        """Execute the step asynchronously."""
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.execute, context)
            return future


# Pipeline class that runs through steps, either sequentially or in parallel
class Pipeline:
    def __init__(self, steps, parallel=False):
        self.steps = steps
        self.parallel = parallel

    def run(self, context):
        if self.parallel:
            self.run_parallel(context)
        else:
            self.run_sequential(context)

    def run_sequential(self, context):
        for step in self.steps:
            step.execute(context)

    def run_parallel(self, context):
        # Create a ThreadPoolExecutor to run steps in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for step in self.steps:
                futures.append(executor.submit(step.execute_async, context))
            for future in as_completed(futures):
                future.result()  # Ensure any exceptions are raised

    



import logging
from typing import List, Dict, Optional

# Assuming the previous classes `Step`, `Config`, and `Artifact` are already defined.

# Logger Setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Data Loading Step
class DataLoadingStep:
    # def __init__(self, config: Optional[Config] = None):
    #     self.config = config or Config()

    @Step(name='DataLoading', function='load_data', inputs=['source'] outputs=["raw_data"])
    def load_data(self, source,config):
        """Load raw data from source."""
        # Simulate loading raw data
        raw_data = "data from source"
        return raw_data


# Data Transformation Step
class DatatransformerStep:
    # def __init__(self, config: Optional[Config] = None):
    #     self.config = config or Config()

    @Step(name='DataTransformer', function='transform_data', inputs=['train_data', 'test_data'], config=config, outputs=['transform_train', 'transform_test'])
    def transform_data(self, train_data, test_data):
        """Logic to transform train and test data."""
        transform_train = f"transformed {train_data}"
        transform_test = f"transformed {test_data}"
        return transform_train, transform_test



# Flowpipe Class to manage the pipeline
class Flowpipe:
    def __init__(self, path: str, config: Config):
        self.path = path
        self.config = config
        self.pipeline = Pipeline([])

    def flow(self):
        """Add steps to the pipeline."""
        data_loading_step = DataLoadingStep()
        data_transformer_step = DatatransformerStep()

        # Add steps to the pipeline
        self.pipeline.steps.append(data_loading_step.load_data)
        self.pipeline.steps.append(data_transformer_step.transform_data)


# Main execution
if __name__ == "__main__":
    path = 'data/data.csv'
    flowpipe_instance = Flowpipe(path)
    flowpipe_instance.flow()
    flowpipe_instance.pipeline.run()



