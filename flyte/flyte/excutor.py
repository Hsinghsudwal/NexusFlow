class Executor:
    def __init__(self, backend='local'):
        self.backend = backend

    def execute(self, task, *args, **kwargs):
        if self.backend == 'local':
            task.run(*args, **kwargs)
        elif self.backend == 'distributed':
            # Distributed task execution logic (e.g., using RabbitMQ or Celery)
            pass