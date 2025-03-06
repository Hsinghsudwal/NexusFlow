import os
import pickle
import hashlib
from datetime import datetime
from typing import Any, Dict

class Artifact:
    def __init__(self, name: str, value: Any, step_id: str, created_at: datetime = None, base_path: str = 'artifacts'):
        """
        Initialize an artifact and manage its storage.

        Args:
            name (str): Name of the artifact
            value (Any): Artifact value/data
            step_id (str): ID of the step that created the artifact
            created_at (datetime, optional): Creation timestamp (defaults to current time if None)
            base_path (str, optional): Directory for storing artifacts (defaults to 'artifacts')
        """
        self.name = name
        self.value = value
        self.step_id = step_id
        self.created_at = created_at or datetime.now()
        self.base_path = base_path
        self.id = self._generate_id()

        # Ensure the base path exists
        os.makedirs(self.base_path, exist_ok=True)

    def _generate_id_artifact(self) -> str:
        """Generate a unique ID for the artifact based on its properties."""
        content = f"{self.name}{self.step_id}{str(self.created_at)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _generate_hash_artifact(self, obj: Any) -> str:
        """Generate a unique hash for the artifact value."""
        serialized = pickle.dumps(obj)
        return hashlib.md5(serialized).hexdigest()

    def save_artifact(self) -> str:
        """
        Save the artifact to the specified directory.

        Returns:
            str: Path where the artifact is saved
        """
        artifact_hash = self._generate_hash_artifact(self.value)
        filename = f"{self.name}_{artifact_hash_artifact}.pkl"
        filepath = os.path.join(self.base_path, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(self.value, f)

        return filepath

    def load_artifact(self, filepath: str) -> Any:
        """
        Load an artifact from the given file path.

        Args:
            filepath (str): Path to the artifact file

        Returns:
            Any: The loaded artifact value
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def list_artifacts(self) -> Dict[str, str]:
        """
        List all stored artifacts in the base path.

        Returns:
            Dict[str, str]: Mapping of artifact filenames to their file paths
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
            name (str): Name of the artifact to retrieve

        Returns:
            Any: The loaded artifact value, or None if not found
        """
        artifact_files = self.list_artifacts()
        for filename, filepath in artifact_files.items():
            if name in filename:
                return self.load(filepath)
        return None  # Return None if the artifact is not found

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
