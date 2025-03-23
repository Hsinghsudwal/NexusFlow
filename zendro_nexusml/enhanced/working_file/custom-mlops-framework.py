# custom_mlops/core/__init__.py

import os
import logging
import yaml
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CustomMLOps")

class MLProject:
    """Main class representing an ML project in the framework"""
    
    def __init__(self, name, description=None):
        """Initialize a new ML project with unique ID and metadata"""
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.components = []
        self.pipelines = []
        self.artifacts = {}
        self.status = "CREATED"
        logger.info(f"Created new ML project: {self.name} with ID: {self.id}")
    
    def add_component(self, component):
        """Add a component to the project"""
        self.components.append(component)
        self.updated_at = datetime.now()
        logger.info(f"Added component {component.name} to project {self.name}")
        return component
    
    def add_pipeline(self, pipeline):
        """Add a pipeline to the project"""
        self.pipelines.append(pipeline)
        self.updated_at = datetime.now()
        logger.info(f"Added pipeline {pipeline.name} to project {self.name}")
        return pipeline
        
    def register_artifact(self, artifact_type, artifact_path, metadata=None):
        """Register an artifact in the project"""
        artifact_id = str(uuid.uuid4())
        self.artifacts[artifact_id] = {
            "type": artifact_type,
            "path": artifact_path,
            "created_at": datetime.now(),
            "metadata": metadata or {}
        }
        self.updated_at = datetime.now()
        logger.info(f"Registered artifact of type {artifact_type} with ID: {artifact_id}")
        return artifact_id
    
    def export_config(self, path=None):
        """Export project configuration as YAML"""
        config = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status,
            "components": [c.to_dict() for c in self.components],
            "pipelines": [p.to_dict() for p in self.pipelines],
            "artifacts": self.artifacts
        }
        
        if path:
            with open(path, 'w') as f:
                yaml.dump(config, f)
            logger.info(f"Exported project configuration to {path}")
            
        return config

class Component:
    """Base class for all ML components in the framework"""
    
    def __init__(self, name, description=None):
        """Initialize a component with metadata"""
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.inputs = {}
        self.outputs = {}
        self.parameters = {}
        self.status = "CREATED"
        self.dependencies = []
        
    def add_input(self, name, data_type, description=None):
        """Add an input specification to the component"""
        self.inputs[name] = {
            "type": data_type,
            "description": description,
            "required": True
        }
        return self
        
    def add_output(self, name, data_type, description=None):
        """Add an output specification to the component"""
        self.outputs[name] = {
            "type": data_type,
            "description": description
        }
        return self
        
    def set_parameter(self, name, value, data_type=None, description=None):
        """Set a parameter for the component"""
        self.parameters[name] = {
            "value": value,
            "type": data_type or type(value).__name__,
            "description": description
        }
        return self
        
    def add_dependency(self, component):
        """Add a dependency to another component"""
        self.dependencies.append(component.id)
        return self
        
    def to_dict(self):
        """Convert component to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "parameters": self.parameters,
            "status": self.status,
            "dependencies": self.dependencies
        }
        
    def run(self, inputs=None):
        """Run the component with the given inputs"""
        raise NotImplementedError("Subclasses must implement the run method")

class Pipeline:
    """Class representing a pipeline of components"""
    
    def __init__(self, name, description=None):
        """Initialize a pipeline with metadata"""
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.steps = []
        self.status = "CREATED"
        self.parameters = {}
        
    def add_step(self, component, name=None, parameters=None):
        """Add a component as a step in the pipeline"""
        step_id = str(uuid.uuid4())
        step = {
            "id": step_id,
            "name": name or f"{component.name}-{len(self.steps)+1}",
            "component_id": component.id,
            "component": component,
            "parameters": parameters or {},
            "status": "PENDING"
        }
        self.steps.append(step)
        return self
        
    def set_parameter(self, name, value, description=None):
        """Set a parameter for the entire pipeline"""
        self.parameters[name] = {
            "value": value,
            "description": description
        }
        return self
        
    def to_dict(self):
        """Convert pipeline to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "steps": [{
                "id": step["id"],
                "name": step["name"],
                "component_id": step["component_id"],
                "parameters": step["parameters"],
                "status": step["status"]
            } for step in self.steps],
            "status": self.status,
            "parameters": self.parameters
        }
        
    def validate(self):
        """Validate the pipeline configuration"""
        # Check for cycles in the dependency graph
        # Validate input/output compatibility between steps
        # More validation logic here
        return True
        
    def execute(self, context=None):
        """Execute the pipeline with the given context"""
        if not self.validate():
            raise ValueError("Pipeline validation failed")
            
        context = context or {}
        self.status = "RUNNING"
        
        try:
            for step in self.steps:
                step["status"] = "RUNNING"
                
                # Merge pipeline parameters with step parameters
                step_params = self.parameters.copy()
                step_params.update(step["parameters"])
                
                # Execute the component
                result = step["component"].run(context)
                
                # Update the context with the results
                context.update(result)
                
                step["status"] = "COMPLETED"
                
            self.status = "COMPLETED"
            return context
            
        except Exception as e:
            self.status = "FAILED"
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
