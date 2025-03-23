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


# Context class to manage the artifacts
class Context:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def update(self, key, artifact):
        self.data[key] = artifact


# Step class that includes configuration and artifact handling
class Step:
    def __init__(self, name, function, inputs=None, outputs=None, config=None):
        self.name = name
        self.function = function
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.config = config or Config()

    def __call__(self, func):
        """This allows the Step class to be used as a decorator."""
        def wrapper(*args, **kwargs):
            # Create the step instance when the function is called
            step_instance = Step(
                name=self.name,
                description=self.description,
                inputs=self.inputs,
                outputs=self.outputs,
                config=self.config,
            )

            # Execute the step
            result = step_instance.execute(func, *args, **kwargs)
            return result

        return wrapper


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
        # Just invoke execute() method in the thread
        return self.execute(context)

    # def execute_async(self, context):
    #     """Execute the step asynchronously."""
    #     with ThreadPoolExecutor() as executor:
    #         future = executor.submit(self.execute, context)
    #         return future


# Pipeline class that runs through steps, either sequentially or in parallel
class Pipeline:
    def __init__(self, steps, parallel=False):
        self.steps = steps
        self.parallel = parallel

    def __call__(self, pipeline_class):
        """This method allows the Pipeline to be used as a decorator."""
        # Add any logic to populate the pipeline's nodes based on the decorated class
        pipeline_class.pipeline = self
        return pipeline_class
        
    def step(name=None, description=None, inputs=None, outputs=None):
        """
        A decorator method for adding steps to the pipeline.
        It takes the step's parameters and appends the step to the pipeline.
        """
        def decorator(func):
            # Create a new step instance with provided parameters
            step = Step(name=name, description=description, inputs=inputs, outputs=outputs)

            def wrapper(*args, **kwargs):
                # Execute the step
                result = step.execute(func, *args, **kwargs)
                return result
            return wrapper
        return decorator

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

    


class DataIngestion:
    # Data Loading Step
    @Step(name='DataLoading', description='Load raw data from source', inputs=['source'], outputs=['train','test'])
    def load_data(source, config):
        """Load raw data from source."""
        # Simulate loading raw data
        return train, test

class DataTransformation:
    # Data Transformation Step
    @Step(name='DataTransformer', description='Transform raw data into usable format', inputs=['train_data', 'test_data'], outputs=['transform_train', 'transform_test'])
    def transform_data(train_data, test_data, config):
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


    # def save_outputs(self, context):
    #     """Save the output artifacts to JSON files."""
    #     for artifact_name, artifact in context.data.items():
    #         output_file = f"{artifact_name}.json"
    #         logger.info(f"Saving artifact: {artifact_name} to {output_file}")
    #         artifact.save(output_file)


    def flow(self):
        """Add steps to the pipeline."""
        my_pipeline = Pipeline(
        name="preprocessing_pipeline",
        description="Pipeline for data preprocessing".
        max_workers=2,
)
        '''step1'''
        dataingestion = DataIngestion()
        train, test = dataingestion.load_data(self.path)
        '''step2'''
        datatrans=DataTransformation()
        transform_train, transform_test = datatrans.transform_data(train, test)

        '''call class'''
        data_ingestion_step = DataIngestion()
        data_transformer_step = DataTransformation()

        my_pipeline.add_step(data_ingestion_step.load_data)
        my_pipeline.add_step(data_transformer_step.transform_data)
        
        # Add steps to the pipeline
        self.pipeline.steps.append(data_ingestion_step.load_data)
        self.pipeline.steps.append(data_transformer_step.transform_data)


# Main execution
if __name__ == "__main__":
    config = Config({'key': 'value'})  # Initialize config (you can load it from a file if needed)
    path = 'data/data.csv'
    flowpipe_instance = Flowpipe(path, config)
    flowpipe_instance.flow()

    # Create a new Context to store and retrieve artifacts
    context = Context()

    # Run the pipeline
    flowpipe_instance.pipeline.run(context)


# Main execution
if __name__ == "__main__":
    config = {'key': 'value'}  # Initialize config (you can load it from a file if needed)
    path = 'data/data.csv'
    flowpipe_instance = Flowpipe(path, config)
    flowpipe_instance.flow()

    # Create a new Context to store and retrieve artifacts
    context = Context()

    # Run the pipeline, executing steps sequentially and passing outputs to the next
    flowpipe_instance.run(context)

    # Save the output artifacts to JSON files
    flowpipe_instance.save_outputs(context)
