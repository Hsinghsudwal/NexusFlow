import yaml
from pathlib import Path

def load_config(project_path: Path) -> Dict:
    """Load project configuration."""
    config_path = project_path / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}
