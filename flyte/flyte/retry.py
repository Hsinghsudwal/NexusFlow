class RetryHandler:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries

    def handle(self, task, *args, **kwargs):
        retries = 0
        while retries < self.max_retries:
            try:
                task.run(*args, **kwargs)
                break
            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    raise e