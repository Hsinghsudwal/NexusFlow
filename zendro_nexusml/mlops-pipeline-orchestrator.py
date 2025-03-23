import os
import json
import uuid
import logging
import datetime
from typing import Dict, List, Any, Optional, Callable
import yaml
import networkx as nx
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mlops-framework")

@dataclass
class ComponentMeta:
    name: str
    version: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
class PipelineComponent:
    """Base class for all pipeline components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.id = str(uuid.uuid4())
        self._meta = self._build_metadata()
        
    def _build_metadata(self) -> ComponentMeta:
        """Build component metadata"""
        return ComponentMeta(
            name=self.__class__.__name__,
            version="0.1.0",
            description="Base pipeline component"
        )
    
    def setup(self) -> None:
        """Prepare resources for execution"""
        logger.info(f"Setting up component {self._meta.name}")
        
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute component logic and return outputs"""
        logger.info(f"Running component {self._meta.name}")
        return {}
        
    def teardown(self) -> None:
        """Clean up resources after execution"""
        logger.info(f"Tearing down component {self._meta.name}")
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """Return component metadata"""
        return asdict(self._meta)
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate inputs against schema"""
        # Implement schema validation logic here
        return True
    
    def validate_outputs(self, outputs: Dict[str, Any]) -> bool:
        """Validate outputs against schema"""
        # Implement schema validation logic here
        return True

@dataclass
class Connection:
    """Connection between pipeline components"""
    upstream: str
    downstream: str
    output_key: Optional[str] = None
    input_key: Optional[str] = None

@dataclass
class PipelineRun:
    """Record of a pipeline execution"""
    id: str
    pipeline_id: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    status: str = "PENDING"
    component_runs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    
class Pipeline:
    """Pipeline definition and execution logic"""
    
    def __init__(self, name: str, description: str = "", version: str = "0.1.0"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.version = version
        self.components: Dict[str, PipelineComponent] = {}
        self.connections: List[Connection] = []
        self.creation_time = datetime.datetime.now()
        self.runs: List[PipelineRun] = []
        
    def add_component(self, name: str, component: PipelineComponent) -> None:
        """Add a component to the pipeline"""
        if name in self.components:
            raise ValueError(f"Component with name '{name}' already exists")
        self.components[name] = component
        
    def connect(self, upstream: str, downstream: str, 
                output_key: str = None, input_key: str = None) -> None:
        """Connect pipeline components"""
        if upstream not in self.components:
            raise ValueError(f"Upstream component '{upstream}' not found")
        if downstream not in self.components:
            raise ValueError(f"Downstream component '{downstream}' not found")
            
        connection = Connection(
            upstream=upstream,
            downstream=downstream,
            output_key=output_key,
            input_key=input_key
        )
        self.connections.append(connection)
    
    def _validate_dag(self) -> bool:
        """Validate that pipeline is a valid DAG"""
        G = nx.DiGraph()
        for name in self.components:
            G.add_node(name)
            
        for conn in self.connections:
            G.add_edge(conn.upstream, conn.downstream)
            
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                logger.error(f"Pipeline contains cycles: {cycles}")
                return False
            return True
        except nx.NetworkXNoCycle:
            return True
    
    def get_execution_order(self) -> List[str]:
        """Return topologically sorted execution order"""
        G = nx.DiGraph()
        for name in self.components:
            G.add_node(name)
            
        for conn in self.connections:
            G.add_edge(conn.upstream, conn.downstream)
            
        return list(nx.topological_sort(G))
    
    def run(self, inputs: Dict[str, Any] = None) -> PipelineRun:
        """Execute the pipeline"""
        if not self._validate_dag():
            raise ValueError("Pipeline is not a valid DAG")
            
        run_id = str(uuid.uuid4())
        run = PipelineRun(
            id=run_id,
            pipeline_id=self.id,
            start_time=datetime.datetime.now(),
            status="RUNNING"
        )
        self.runs.append(run)
        
        execution_order = self.get_execution_order()
        component_outputs = inputs or {}
        
        try:
            for component_name in execution_order:
                component = self.components[component_name]
                
                # Set up component
                component.setup()
                
                # Collect inputs for this component
                component_inputs = {}
                for conn in self.connections:
                    if conn.downstream == component_name and conn.upstream in component_outputs:
                        if conn.output_key is not None and conn.input_key is not None:
                            # Connect specific outputs to inputs
                            component_inputs[conn.input_key] = component_outputs[conn.upstream].get(conn.output_key)
                        elif conn.output_key is None and conn.input_key is None:
                            # Connect all outputs to inputs
                            component_inputs.update(component_outputs[conn.upstream])
                
                # Add any global inputs that might be relevant
                if inputs and component_name in inputs:
                    component_inputs.update(inputs[component_name])
                
                # Execute component
                start_time = datetime.datetime.now()
                outputs = component.run(component_inputs)
                end_time = datetime.datetime.now()
                
                # Store outputs
                component_outputs[component_name] = outputs
                
                # Record component run
                run.component_runs[component_name] = {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "status": "COMPLETED",
                    "inputs": component_inputs,
                    "outputs": outputs
                }
                
                # Clean up
                component.teardown()
            
            # Mark run as completed
            run.status = "COMPLETED"
            run.end_time = datetime.datetime.now()
            
            # Store final outputs as artifacts
            run.artifacts = component_outputs
            
            return run
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            run.status = "FAILED"
            run.end_time = datetime.datetime.now()
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "creation_time": self.creation_time.isoformat(),
            "components": {
                name: component.metadata for name, component in self.components.items()
            },
            "connections": [asdict(conn) for conn in self.connections]
        }
    
    def save(self, path: str) -> None:
        """Save pipeline definition to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, path: str) -> 'Pipeline':
        """Load pipeline from disk (partial implementation)"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        pipeline = cls(
            name=data["name"],
            description=data["description"],
            version=data["version"]
        )
        pipeline.id = data["id"]
        
        # NOTE: This only loads the structure, not the actual component implementations
        # In a real implementation, you would need a component registry
        
        return pipeline

class MLOpsFramework:
    """Main entry point for the MLOps framework"""
    
    def __init__(self, storage_dir: str = "./mlops_storage"):
        self.storage_dir = storage_dir
        self.pipelines: Dict[str, Pipeline] = {}
        
        # Ensure storage directories exist
        os.makedirs(os.path.join(storage_dir, "pipelines"), exist_ok=True)
        os.makedirs(os.path.join(storage_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(storage_dir, "datasets"), exist_ok=True)
        os.makedirs(os.path.join(storage_dir, "runs"), exist_ok=True)
        
    def create_pipeline(self, name: str, description: str = "") -> Pipeline:
        """Create a new pipeline"""
        pipeline = Pipeline(name=name, description=description)
        self.pipelines[pipeline.id] = pipeline
        return pipeline
    
    def save_pipeline(self, pipeline_id: str) -> None:
        """Save pipeline definition to disk"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline with id '{pipeline_id}' not found")
            
        pipeline = self.pipelines[pipeline_id]
        path = os.path.join(self.storage_dir, "pipelines", f"{pipeline_id}.yaml")
        pipeline.save(path)
    
    def load_pipeline(self, pipeline_id: str) -> Pipeline:
        """Load pipeline from disk"""
        path = os.path.join(self.storage_dir, "pipelines", f"{pipeline_id}.yaml")
        pipeline = Pipeline.load(path)
        self.pipelines[pipeline.id] = pipeline
        return pipeline
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipelines"""
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "version": p.version,
                "created": p.creation_time.isoformat()
            }
            for p in self.pipelines.values()
        ]
    
    def run_pipeline(self, pipeline_id: str, inputs: Dict[str, Any] = None) -> PipelineRun:
        """Run a pipeline"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline with id '{pipeline_id}' not found")
            
        pipeline = self.pipelines[pipeline_id]
        run = pipeline.run(inputs)
        
        # Save run details
        run_path = os.path.join(self.storage_dir, "runs", f"{run.id}.json")
        with open(run_path, 'w') as f:
            json.dump(asdict(run), f, default=str)
            
        return run
    
    def get_pipeline_run(self, run_id: str) -> PipelineRun:
        """Get details of a pipeline run"""
        run_path = os.path.join(self.storage_dir, "runs", f"{run_id}.json")
        with open(run_path, 'r') as f:
            data = json.load(f)
        
        # Convert dictionary back to PipelineRun
        run = PipelineRun(
            id=data["id"],
            pipeline_id=data["pipeline_id"],
            start_time=datetime.datetime.fromisoformat(data["start_time"]),
            status=data["status"],
            component_runs=data["component_runs"],
            artifacts=data["artifacts"]
        )
        
        if data["end_time"]:
            run.end_time = datetime.datetime.fromisoformat(data["end_time"])
            
        return run
