# tests/test_pipeline.py
from pipelines.training_pipeline import training_pipeline
from core.config import load_config

def test_training_pipeline_runs():
    config = load_config("local")
    result = training_pipeline(config)
    assert result.get("model") is not None
