
import logging
import json_log_formatter

class JSONLogger:
    def __init__(self, name="nexusflow"):
        self.logger = logging.getLogger(name)
        self.formatter = json_log_formatter.JSONFormatter()
        
    def log_pipeline_start(self, pipeline_name):
        self.logger.info({
            "event": "pipeline_start",
            "pipeline": pipeline_name,
            "status": "running"
        })
        
    def log_step_execution(self, node_name, duration, success=True):
        self.logger.info({
            "event": "node_execution",
            "node": node_name,
            "duration": duration,
            "success": success
        }, exc_info=not success)