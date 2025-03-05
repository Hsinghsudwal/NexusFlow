class Workflow:
    def __init__(self, name):
        self.name = name
        self.tasks = []
        self.dependencies = {}  # Map task -> list of dependent tasks

    def add_task(self, task, dependencies=[]):
        self.tasks.append(task)
        self.dependencies[task] = dependencies

    def run(self):
        # Logic to execute tasks in order considering dependencies
        for task in self.tasks:
            if self.dependencies[task] == []:  # Task with no dependencies
                task.run()
            else:
                # Handle task with dependencies
                pass