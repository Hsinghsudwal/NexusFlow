from typing import List, Optional, Set
from .node import Node


class Pipeline:
    """
    A pipeline is a collection of nodes that are executed in order.
    """
    def __init__(
        self,
        nodes: List[Node],
        name: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ):
        self.nodes = nodes
        self.name = name or "pipeline"
        self.tags = tags or set()
        self._validate_pipeline()

    def _validate_pipeline(self):
        """Validate that the pipeline is well-formed."""
        all_inputs = set()
        all_outputs = set()

        for node in self.nodes:
            all_inputs.update(node.inputs.values())
            all_outputs.update(node.outputs.values())

        missing_inputs = all_inputs - all_outputs

    def only_nodes_with_tags(self, tags: Set[str]) -> "Pipeline":
        """Return a new pipeline with only nodes that have the specified tags."""
        nodes = [node for node in self.nodes if tags.intersection(node.tags)]
        return Pipeline(nodes=nodes, name=self.name)

    def __add__(self, other: "Pipeline") -> "Pipeline":
        """Combine two pipelines."""
        if not isinstance(other, Pipeline):
            return NotImplemented

        return Pipeline(
            nodes=self.nodes + other.nodes,
            name=f"{self.name}+{other.name}"
        )



    
    def __init__(self, name: str, steps: Optional[List[BaseStep]] = None):
        self.name = name
        self.steps = steps or []
        self.uuid = str(uuid.uuid4())
        self._step_dependencies = {}
        
    def add_step(self, step: BaseStep) -> "Pipeline":
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self
        
    def run(self, context):
        """Run the pipeline with a given context."""
        results = {}
        
        # In ZenML-like systems, the orchestrator would handle this
        # This is a simple sequential execution
        for step in self.steps:
            # Prepare inputs from previous steps if needed
            kwargs = {}
            for param, artifact_key in step._input_artifacts.items():
                if artifact_key in results:
                    kwargs[param] = results[artifact_key]
                    
            # Execute the step
            result = step(**kwargs)
            
            # Store the result
            results[step.name] = result
            
        return results