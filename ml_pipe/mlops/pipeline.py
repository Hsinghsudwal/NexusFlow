import yaml
import importlib
from prefect import flow, task

@task
def run_step(step_config, context):
    module = importlib.import_module(step_config["module"])
    function = getattr(module, step_config["function"])
    return function(**step_config.get("parameters", {}))

@flow
def run_pipeline(pipeline_config):
    context = {}
    for step in pipeline_config["steps"]:
        context[step["name"]] = run_step(step, context)
    return context

if __name__ == "__main__":
    with open("pipeline.yaml", "r") as f:
        pipeline_config = yaml.safe_load(f)
    run_pipeline(pipeline_config)