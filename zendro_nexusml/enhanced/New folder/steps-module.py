# mlops_framework/steps.py

import inspect
import os
import uuid
import pickle
import datetime
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

from .metadata_store import MetadataStore
from .artifact_store import ArtifactStore
from .config import ConfigManager

logger = logging.getLogger(__name__)

def step(name: Optional[str] = None, cache: bool = True, artifacts: List[str] = None):
    """
    Decorator to define a pipeline step.
    
    Args:
        name: Optional custom name for the step
        cache: Whether to cache the step's output
        artifacts: List of artifact names to save from the step output
    
    Returns:
        Decorated function
    """
    def decorator(func):
        step_id = str(uuid.uuid4())
        step_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Starting step '{step_name}'")
            
            # Access the global instances
            metadata_store = MetadataStore()
            artifact_store = ArtifactStore()
            config = ConfigManager()
            
            # Check if we can use cached result
            if cache:
                # Create a cache key based on function name, args, and kwargs
                import hashlib
                cache_key = f"{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}"
                cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
                
                cached_result = metadata_store.get_cached_result(cache_hash)
                if cached_result:
                    logger.info(f"Using cached result for step '{step_name}'")
                    return cached_result
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Save artifacts if specified
            if artifacts and isinstance(result, dict):
                for artifact_name in artifacts:
                    if artifact_name in result:
                        artifact_store.save_artifact(
                            name=artifact_name,
                            data=result[artifact_name],
                            metadata={
                                'step_name': step_name,
                                'timestamp': datetime.datetime.now().isoformat()
                            }
                        )
            
            # Cache the result if enabled
            if cache:
                metadata_store.cache_result(cache_hash, result)
            
            logger.info(f"Completed step '{step_name}'")
            return result
        
        # Store step metadata on the function
        wrapper.step_id = step_id
        wrapper.step_name = step_name
        wrapper.is_step = True
        
        return wrapper
    
    return decorator
