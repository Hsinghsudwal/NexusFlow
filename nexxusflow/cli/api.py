# runners/api.py
from fastapi import FastAPI, Request
from pipelines.training_pipeline import training_pipeline
from rollback.rollback_manager import RollbackManager
from feedback.feedback_loop import FeedbackLoop
from core.config import load_config

app = FastAPI()
config = load_config("local")  # or read from env var

@app.post("/run/training")
def run_training():
    training_pipeline(config)
    return {"status": "Training pipeline triggered"}

@app.post("/rollback/{model_key}")
def rollback_model(model_key: str):
    manager = RollbackManager(config)
    model = manager.rollback(model_key)
    return {"status": "Rolled back", "model": model}

@app.post("/feedback")
async def submit_feedback(request: Request):
    data = await request.json()
    feedback = FeedbackLoop(config)
    feedback.collect_feedback(str(data))
    return {"status": "Feedback collected"}
