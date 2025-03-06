import importlib
import yaml
from .pipeline import Pipeline
from .catalog import DataCatalog


class ProjectContext:
    """Context object for managing project resources."""
    
    def __init__(self, project_path: Union[str, Path]):
        self.project_path = Path(project_path)
        self.config = self._load_config()
        self.catalog = DataCatalog(self.project_path / self.config.get('data_path', 'data'))
        
    def _load_config(self) -> Dict:
        """Load project configuration."""
        config_path = self.project_path / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def load_pipeline(self, module_name: str, pipeline_name: str = 'pipeline') -> Pipeline:
        """Load a pipeline from a Python module."""
        try:
            module = importlib.import_module(module_name)
            pipeline = getattr(module, pipeline_name)
            if not isinstance(pipeline, Pipeline):
                raise TypeError(f"Object '{pipeline_name}' in module '{module_name}' is not a Pipeline")
            return pipeline
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to load pipeline '{pipeline_name}' from module '{module_name}': {str(e)}")
            
    def run_pipeline(self, pipeline: Union[Pipeline, str], pipeline_name: str = 'pipeline', **kwargs) -> Dict[str, Any]:
        """Run a pipeline with the project context."""
        if isinstance(pipeline, str):
            pipeline = self.load_pipeline(pipeline, pipeline_name)
        return pipeline.run(self.catalog, **kwargs)
