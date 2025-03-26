# class ArtifactStore:
#     """Stores and retrieves intermediate artifacts for the pipeline."""

#     def __init__(self, config):
#         self.config = config
#         self.base_path = self.config.get("folder_path", {}).get(
#             "artifacts", "artifacts"
#         )
#         os.makedirs(self.base_path, exist_ok=True)
#         logging.info(f"Artifact store initialized at '{self.base_path}'")

#     def save_artifact(
#         self,
#         artifact: Any,
#         subdir: str,
#         name: str,
#     ) -> str:
#         """Save an artifact in the specified format and return the path."""
#         artifact_dir = os.path.join(self.base_path, subdir)
#         os.makedirs(artifact_dir, exist_ok=True)
#         artifact_path = os.path.join(artifact_dir, name)

#         if name.endswith(".pkl"):
#             with open(artifact_path, "wb") as f:
#                 pickle.dump(artifact, f)
#         elif name.endswith(".csv"):
#             if isinstance(artifact, pd.DataFrame):
#                 artifact.to_csv(artifact_path, index=False)
#             else:
#                 raise ValueError("CSV format only supports pandas DataFrames.")
#         else:
#             raise ValueError(f"Unsupported format for {name}")
#         logging.info(f"Artifact '{name}' saved to {artifact_path}")
#         return artifact_path

#     def load_artifact(
#         self,
#         subdir: str,
#         name: str,
#     ):
#         """Load an artifact in the specified format."""
#         artifact_path = os.path.join(self.base_path, subdir, name)
#         if os.path.exists(artifact_path):
#             if name.endswith(".pkl"):
#                 with open(artifact_path, "rb") as f:
#                     artifact = pickle.load(f)
#             elif name.endswith(".csv"):
#                 artifact = pd.read_csv(artifact_path)
#             else:
#                 raise ValueError(f"Unsupported format for {name}")
#             logging.info(f"Artifact '{name}' loaded from {artifact_path}")
#             return artifact
#         else:
#             logging.warning(f"Artifact '{name}' not found in {artifact_path}")
#             return None
            
#     def list_artifacts(self, run_id=None):
#         """List all artifacts or artifacts for a specific run."""
#         artifacts = []
#         for root, _, files in os.walk(self.base_path):
#             for file in files:
#                 artifact_path = os.path.join(root, file)
#                 # If run_id is specified, only include artifacts containing that run_id
#                 if run_id is None or run_id in artifact_path:
#                     artifacts.append(artifact_path)
#         return artifacts
