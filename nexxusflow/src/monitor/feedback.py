# feedback_loop.py
from core.artifact_manager import ArtifactManager

class FeedbackLoop:
    def __init__(self, config):
        self.artifacts = ArtifactManager(config)

    def collect_feedback(self, data):
        self.artifacts.save("feedback/latest.json", data)

    def trigger_retraining(self):
        # Read feedback and rerun pipeline
        print("Triggering retraining with new feedback...")
