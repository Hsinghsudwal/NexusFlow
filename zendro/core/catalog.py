

import os
import pickle
import yaml
import logging
import datetime
import importlib
import inspect
import hashlib
from typing import Any, Dict, List, Callable, Optional, Union, Set, Tuple
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DataCatalog:
    """Data catalog for managing dataset storage and retrieval."""
    base_path: Path

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, data: Any, metadata: Optional[Dict] = None) -> None:
        """Save data to the catalog."""
        data_path = self.base_path / f"{name}.pkl"
        metadata_path = self.base_path / f"{name}.meta.yaml"
        
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save metadata if provided
        if metadata:
            current_metadata = {
                'created_at': datetime.datetime.now().isoformat(),
                'data_hash': hashlib.md5(pickle.dumps(data)).hexdigest()
            }
            current_metadata.update(metadata)
            
            with open(metadata_path, 'w') as f:
                yaml.dump(current_metadata, f)
                
        logger.info(f"Saved data '{name}' to {data_path}")

    def load(self, name: str) -> Any:
        """Load data from the catalog."""
        data_path = self.base_path / f"{name}.pkl"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data '{name}' not found in catalog")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        logger.info(f"Loaded data '{name}' from {data_path}")
        return data
    
    def exists(self, name: str) -> bool:
        """Check if data exists in the catalog."""
        data_path = self.base_path / f"{name}.pkl"
        return data_path.exists()
    
    def get_metadata(self, name: str) -> Dict:
        """Get metadata for a dataset."""
        metadata_path = self.base_path / f"{name}.meta.yaml"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata for '{name}' not found in catalog")
        
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f)