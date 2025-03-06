import logging

class Logger:
    def __init__(self, log_file="experiment.log"):
        self.logger = logging.getLogger("NexusML")
        self.logger.setLevel(logging.INFO)
        self.handler = logging.FileHandler(log_file)
        self.logger.addHandler(self.handler)
        
    def log(self, message):
        self.logger.info(message)
