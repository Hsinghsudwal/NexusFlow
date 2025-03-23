import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import uuid
from .step import Step
from .experiment import Experiment
import logging

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, name: str, description: str = None):
        self.name = name
        self.description = description
        self.steps = {}
        self.dag = nx.DiGraph()
        self.id = str(uuid.uuid4())

    def add_step(self, step: Step, dependencies: list = None):
        self.steps[step.name] = step
        self.dag.add_node(step.name)

        if dependencies:
            for dep in dependencies:
                self.dag.add_edge(dep.name, step.name)

        # Verify acyclic
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("Pipeline steps contain a cycle")

    def run(self, config=None):
        """Execute the pipeline steps in parallel or sequentially."""
        # Create experiment
        experiment = Experiment(pipeline_id=self.id, pipeline_name=self.name, start_time=datetime.now())

        # Start parallel execution of steps
        with ThreadPoolExecutor() as executor:
            futures = {}
            for step_name in nx.topological_sort(self.dag):
                step = self.steps[step_name]
                input_artifacts = {}

                for dep in self.dag.predecessors(step_name):
                    for key, artifact in self.steps[dep].artifacts.items():
                        input_artifacts[key] = artifact.value

                futures[step_name] = executor.submit(step.run, input_artifacts, config)

            # Wait for all futures to complete
            for future in futures.values():
                future.result()

        experiment.status = "completed"
        experiment.end_time = datetime.now()
        experiment.save()
