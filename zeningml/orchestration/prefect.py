from prefect import flow, task

class PrefectOrchestrator:
    def __init__(self, stack):
        self.stack = stack

    def create_flow(self, pipeline):
        @flow
        def prefect_flow():
            for step in pipeline.steps:
                self.execute_step(step)
        return prefect_flow

    def execute_step(self, step):
        @task
        def step_task():
            return step.execute()
        return step_task()