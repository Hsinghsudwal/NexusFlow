import abc
import requests
from datetime import datetime
from typing import Any, Dict
import json
import boto3
import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, List, Optional
import datetime
import inspect
import graphviz
from concurrent.futures import ThreadPoolExecutor, as_completed


# 🔹 Artifact Class for Storing and Managing Data
class Artifact:
    def __init__(self, name: str, data: Any = None):
        self.name = name
        self.data = data

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.data, f)

    def load(self, path: str):
        with open(path, 'r') as f:
            self.data = json.load(f)

    def upload(self, store, path: str):
        store.save(self, path)

    def download(self, store, path: str):
        return store.load(path)

    def __repr__(self):
        return f"Artifact(name={self.name}, data={self.data})"


# 🔹 Context Class to Handle Artifacts During Execution
class Context:
    def __init__(self):
        self.data = {}

    def get(self, key: str):
        return self.data.get(key)

    def update(self, key: str, artifact: Artifact):
        self.data[key] = artifact
        print(f"Artifact saved: {artifact.name}, data: {artifact.data}")


# 🔹 Config Class for Loading and Managing Pipeline Configuration
class Config:
    def __init__(self, config_dict: Dict = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None):
        return self.config_dict.get(key, default)

    @staticmethod
    def load_from_file(filename: str):
        with open(filename, 'r') as file:
            config_data = json.load(file)
        return Config(config_data)


# 🔹 Pipeline Registry to Manage Pipeline Versions
class PipelineRegistry:
    def __init__(self, storage_backend: 'ArtifactStore', registry_path: str = "registered_pipelines.pkl"):
        self.storage = storage_backend
        self.pipelines = {}
        self.registry_path = registry_path
        self._load_registry_from_file()

    def _load_registry_from_file(self):
        """Load the pipeline registry from a file if it exists."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "rb") as file:
                    self.pipelines = pickle.load(file)
                logger.info("Pipeline registry loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading pipeline registry: {e}")

    def register_pipeline(self, pipeline: 'Pipeline'):
        """Register the pipeline by saving its metadata to both in-memory registry and file."""
        try:
            # Register pipeline in memory
            pipeline_id = f"{pipeline.name}-{pipeline.config.get('version', '1.0.0')}"
            self.pipelines[pipeline_id] = {
                'metadata': pipeline.execution_metadata,
                'created_at': datetime.datetime.now(),
                'artifacts': []
            }

            # Save pipeline metadata to file
            self._save_registry_to_file()

            # Optionally save artifacts to the storage backend
            self._save_to_storage(pipeline_id)

            logger.info(f"Pipeline '{pipeline.name}' registered successfully.")

        except Exception as e:
            logger.error(f"Error registering pipeline '{pipeline.name}': {e}")
            raise

    def _save_registry_to_file(self):
        """Save the entire registry to a file."""
        try:
            with open(self.registry_path, "wb") as file:
                pickle.dump(self.pipelines, file)
            logger.info(f"Pipeline registry saved to {self.registry_path}.")
        except Exception as e:
            logger.error(f"Error saving pipeline registry: {e}")
            raise

    def _save_to_storage(self, pipeline_id: str):
        """Save the pipeline's artifacts to the storage backend."""
        try:
            self.storage.save(Artifact(name=pipeline_id, data=self.pipelines[pipeline_id]), pipeline_id)
        except Exception as e:
            logger.error(f"Error saving artifacts for pipeline '{pipeline_id}': {e}")
            raise

# 🔹 Model Registry Class to Handle Model Versioning and Promotion
class ModelRegistry:
    def get_latest_production_model(self):
        """Fetch the latest production model."""
        # For simplicity, this is a placeholder
        return "previous_model"

    def promote_model(self, model_path: str, stage: str):
        """Promote a model to the given stage (e.g., production)."""
        print(f"Model '{model_path}' promoted to {stage}.")











