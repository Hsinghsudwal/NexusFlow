class Versioning:
    def __init__(self):
        self.version_history = {}

    def version(self, name, version_data):
        self.version_history[name] = version_data
        print(f"Versioned {name} with data: {version_data}")

     def get_version(self, name: str) -> str:
        """
        Retrieve the version data for a specific item.

        Args:
            name (str): The name of the item whose version data is requested.

        Returns:
            str: The version data for the item.
        """
        return self._version_history.get(name, "Version data not found")

    def list_versions(self) -> dict:
        """List all versioned items and their data."""
        return self._version_history

# Example usage:

# Initialize versioning system
# versioning = Versioning()

# Version a model
# versioning.version("model_v1", "Initial model version")

# Get the version data of an item
# print(versioning.get_version("model_v1"))

# List all versioned items
# print(versioning.list_versions())