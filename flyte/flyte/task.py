class Task:
    def __init__(self, func):
        self.func = func
        self.metadata = {}  # Store task-specific metadata (e.g., input/output types)

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)
       
    def set_metadata(self, key, value):
        self.metadata[key] = value




def task(func):
    return Task(func)