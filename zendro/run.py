# Import the framework
from datapipe import create_project, ProjectContext, node, Pipeline

# Create a new project
create_project("my_project")

# Create a node with a decorator
@node(outputs="my_data")
def generate_data():
    return {"value": 42}

# Run a pipeline
context = ProjectContext("my_project")
context.run_pipeline("src.pipelines.example")