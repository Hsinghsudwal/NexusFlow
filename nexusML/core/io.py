class Stack:
    def __init__(self, name: str, max_workers: int = 4):
        self.name = name
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks = []

    def add_task(self, fn, *args, **kwargs):
        self.tasks.append(self.executor.submit(fn, *args, **kwargs))

    def run(self) -> Dict:
        results = {}
        for future in self.tasks:
            task_name, result = future.result()
            results[task_name] = result
        return results