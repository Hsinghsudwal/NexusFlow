# training_pipeline.py
from core.pipeline import Step, Pipeline

def load_data(context):
    print("Loading data...")
    return {"data": [1, 2, 3, 4]}

def train_model(context):
    data = context.get("data", [])
    print("Training model on:", data)
    context["model"] = f"model_trained_on_{len(data)}_samples"
    return context

def training_pipeline(config):
    steps = [
        Step("load_data", load_data),
        Step("train_model", train_model),
    ]
    pipeline = Pipeline("training", steps)
    context = {"config": config}
    return pipeline.run(context)
