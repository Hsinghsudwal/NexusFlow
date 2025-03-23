from setuptools import setup, find_packages

setup(
    name="mlops_framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'mlops-run=mlops_framework.runner:main',
        ],
    },
    author="Your Name",
    description="A minimal MLOps framework similar to ZenML",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


mlops_framework/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── storage.py  # Artifact storage
│   ├── step.py     # Step abstraction
│   ├── pipeline.py # Pipeline execution
├── steps/
│   ├── __init__.py
│   ├── load_data.py
│   ├── process_data.py
│   ├── train_model.py
├── runner.py       # Main pipeline runner
└── setup.py        # Package installation setup

# storage.py
import os
import pickle






# load_data.py
def load_data():
    return [1, 2, 3, 4, 5]

# process_data.py
def process_data(data):
    return [x * 2 for x in data]

# train_model.py
def train_model(processed_data):
    return sum(processed_data) / len(processed_data)  # Mock model

# runner.py
from mlops_framework.core.pipeline import Pipeline
from mlops_framework.core.step import Step
from mlops_framework.steps.load_data import load_data
from mlops_framework.steps.process_data import process_data
from mlops_framework.steps.train_model import train_model

# Define steps
load_step = Step("load_data", load_data)
process_step = Step("process_data", process_data, dependencies=["load_data"])
train_step = Step("train_model", train_model, dependencies=["process_data"])

# Run pipeline
if __name__ == "__main__":
    pipeline = Pipeline([load_step, process_step, train_step])
    pipeline.run()

from steps.load_data import LoadData
from steps.process_data import ProcessData
from steps.train_model import TrainModel
from pipeline import Pipeline

# Define steps
steps = [LoadData(), ProcessData(), TrainModel()]

# Run sequentially
pipeline = Pipeline(steps, parallel=False)
pipeline.run()

# Run in parallel
pipeline_parallel = Pipeline(steps, parallel=True, max_workers=3)
pipeline_parallel.run()


from steps.load_data import LoadData
from steps.process_data import ProcessData
from steps.train_model import TrainModel
from pipeline import Pipeline

# Define steps
steps = [LoadData(), ProcessData(), TrainModel()]

# Run sequentially
pipeline = Pipeline(steps, parallel=False)
pipeline.run()

# Run in parallel (max_workers is automatically handled)
pipeline_parallel = Pipeline(steps, parallel=True)
pipeline_parallel.run()



