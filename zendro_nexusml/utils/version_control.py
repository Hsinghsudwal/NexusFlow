# artifact_management/version_control.py

import os

class VersionControl:
    def __init__(self, base_path):
        self.base_path = base_path
    
    def save_version(self, name, content):
        versioned_path = os.path.join(self.base_path, f"{name}_v1")
        with open(versioned_path, "w") as f:
            f.write(content)
    
    def load_version(self, name):
        versioned_path = os.path.join(self.base_path, f"{name}_v1")
        with open(versioned_path, "r") as f:
            return f.read()
