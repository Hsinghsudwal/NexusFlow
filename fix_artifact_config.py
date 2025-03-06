import os
import pickle
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Artifact:
    def __init__(self, name: str, value: Any, step_id: str, base_path: str = 'artifacts', created_at: datetime = None):
        """
        Initialize an artifact and handle its storage operations.

        Args:
            name (str): Name of the artifact.
            value (Any): Value or data of the artifact.
            step_id (str): The ID of the process that created the artifact.
            created_at (datetime, optional): Creation timestamp (defaults to current time if None).
            base_path (str, optional): Directory to store artifacts (defaults to 'artifacts').
        """
        self.name = name
        self.value = value
        self.step_id = step_id
        self.created_at = created_at or datetime.now()
        self.base_path = base_path
        self.id = self._generate_id()

        # Ensure the base path exists
        os.makedirs(self.base_path, exist_ok=True)

    def _generate_id(self) -> str:
        """Generate a unique ID for the artifact based on its properties."""
        content = f"{self.name}{self.step_id}{str(self.created_at)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _generate_hash(self) -> str:
        """Generate a unique hash for the artifact's value."""
        serialized = pickle.dumps(self.value)
        return hashlib.md5(serialized).hexdigest()

    def save(self) -> str:
        """
        Save the artifact to the specified directory.

        Returns:
            str: Path where the artifact is saved.
        """
        artifact_hash = self._generate_hash()
        filename = f"{self.name}_{artifact_hash}.pkl"
        filepath = os.path.join(self.base_path, filename)

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.value, f)
            logger.info(f"Artifact '{self.name}' saved at {filepath}")
        except Exception as e:
            logger.error(f"Error saving artifact '{self.name}': {e}")
            raise
        
        return filepath

    def load(self, filepath: str) -> Any:
        """
        Load an artifact from the given file path.

        Args:
            filepath (str): Path to the artifact file.

        Returns:
            Any: The loaded artifact value.
        """
        try:
            with open(filepath, 'rb') as f:
                artifact_value = pickle.load(f)
            logger.info(f"Artifact loaded from {filepath}")
            return artifact_value
        except Exception as e:
            logger.error(f"Error loading artifact from {filepath}: {e}")
            raise

    def list_artifacts(self) -> Dict[str, str]:
        """
        List all artifacts in the base path.

        Returns:
            Dict[str, str]: Mapping of artifact filenames to their file paths.
        """
        return {
            f: os.path.join(self.base_path, f)
            for f in os.listdir(self.base_path)
            if f.endswith('.pkl')
        }

    def get_artifact(self, name: str) -> Any:
        """
        Retrieve an artifact by its name.

        Args:
            name (str): The name of the artifact to retrieve.

        Returns:
            Any: The loaded artifact value, or None if not found.
        """
        artifact_files = self.list_artifacts()
        for filename, filepath in artifact_files.items():
            if name in filename:
                return self.load(filepath)
        logger.error(f"Artifact '{name}' not found.")
        return None  # Return None if the artifact is not found

    def delete(self, artifact_name: str):
        """
        Delete an artifact from the artifact directory.

        Args:
            artifact_name (str): Name of the artifact to delete.
        """
        artifact_path = Path(self.base_path) / artifact_name
        if artifact_path.exists():
            try:
                artifact_path.unlink()  # Delete the artifact
                logger.info(f"Artifact '{artifact_name}' deleted.")
            except Exception as e:
                logger.error(f"Error deleting artifact '{artifact_name}': {e}")
                raise
        else:
            logger.error(f"Artifact '{artifact_name}' not found, cannot delete.")
            raise FileNotFoundError(f"Artifact '{artifact_name}' not found to delete.")

    def clear_all_artifacts(self):
        """
        Clear all artifacts stored in the artifact directory.
        """
        try:
            for artifact in Path(self.base_path).iterdir():
                if artifact.is_file():
                    artifact.unlink()  # Delete each artifact file
            logger.info("All artifacts cleared.")
        except Exception as e:
            logger.error(f"Error clearing artifacts: {e}")
            raise

    def __repr__(self):
        return f"Artifact(name={self.name}, step_id={self.step_id}, created_at={self.created_at}, id={self.id})"

# Example usage:

# Create an artifact
# artifact = Artifact(name="model", value="model_data_placeholder", step_id="training_step")

# Save the artifact
# saved_path = artifact.save()
# print(f"Artifact saved at: {saved_path}")

# List all artifacts in the store
# stored_artifacts = artifact.list_artifacts()
# print(f"Stored artifacts: {stored_artifacts}")

# Retrieve an artifact by name
# retrieved_artifact = artifact.get_artifact("model")
# print(f"Retrieved artifact value: {retrieved_artifact}")

# Delete the artifact
# artifact.delete("model_123456.pkl")  # Adjust the name based on actual artifact hash

# Clear all artifacts
# artifact.clear_all_artifacts()


import os
import yaml
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager class to handle loading, saving, and accessing configuration values."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager and optionally load configuration.
        
        Args:
            config_path (str, optional): Path to YAML config file.
        """
        self.config: Dict[str, Any] = {}
        if config_path:
            self.load(config_path)

    def load(self, path: str) -> None:
        """Load configuration from a YAML file.
        
        Args:
            path (str): Path to the YAML config file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            self.config = yaml.safe_load(f) or {}
        logger.info(f"Configuration loaded from {path}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value for a specific key.
        
        Args:
            key (str): The key of the configuration item.
            default (Any): The default value if the key is not found.

        Returns:
            Any: The configuration value for the given key.
        """
        value = self.config.get(key, default)
        logger.debug(f"Config: {key} = {value}")
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key (str): Configuration key.
            value (Any): Value to set.
        """
        self.config[key] = value
        logger.info(f"Config set: {key} = {value}")

    def save(self, path: str) -> None:
        """Save the current configuration to a YAML file.
        
        Args:
            path (str): The path to save the configuration file.
        """
        with open(path, "w") as f:
            yaml.dump(self.config, f)
        logger.info(f"Configuration saved to {path}")

    def update(self, key: str, value: Any) -> None:
        """Update configuration value.
        
        Args:
            key (str): Configuration key.
            value (Any): Value to update.
        """
        self.config[key] = value
        logger.info(f"Config updated: {key} = {value}")
    
    def list(self) -> Dict[str, Any]:
        """List all configurations.
        
        Returns:
            Dict[str, Any]: All configurations stored in the manager.
        """
        return self.config

    def clear(self) -> None:
        """Clear all configurations in memory."""
        self.config.clear()
        logger.info("All configuration data cleared.")
    

# Example Usage:
# Initialize the ConfigManager with a path to the configuration file
# config_manager = ConfigManager(config_path="config.yaml")

# Get a configuration value
# pipeline_name = config_manager.get("pipeline_name", default="default_pipeline")
# print(f"Pipeline Name: {pipeline_name}")

# Set and save new configuration value
# config_manager.set("model_version", "v1.0")
# config_manager.save("config.yaml")

# List all configurations
# all_configs = config_manager.list()
# print(f"All Configurations: {all_configs}")

# Clear all configurations from memory
# config_manager.clear()
