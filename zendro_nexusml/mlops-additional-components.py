# mlops_framework/tracking/tracker.py
import os
import json
import logging
import time
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Tracker for model metrics and experiment results."""
    
    def __init__(self, log_dir: str = "./logs"):
        """
        Initialize a MetricsTracker.
        
        Args:
            log_dir: Directory for storing logs
        """
        self.log_dir = Path(log_dir)
        self.metrics_dir = self.log_dir / "metrics"
        self.artifacts_dir = self.log_dir / "artifacts"
        self.runs_dir = self.log_dir / "runs"
        
        # Create directories
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self._metrics = {}
    
    def log_metric(
        self,
        name: str,
        value: float,
        run_id: Optional[str] = None,
        step: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Log a numeric metric.
        
        Args:
            name: Metric name
            value: Metric value
            run_id: Run identifier
            step: Training step or epoch
            timestamp: Timestamp for the metric
        """
        timestamp = timestamp or datetime.datetime.now().isoformat()
        run_id = run_id or "default"
        
        if run_id not in self._metrics:
            self._metrics[run_id] = {}
        
        if name not in self._metrics[run_id]:
            self._metrics[run_id][name] = []
        
        metric_value = {
            "value": float(value),
            "timestamp": timestamp,
        }
        
        if step is not None:
            metric_value["step"] = step
        
        self._metrics[run_id][name].append(metric_value)
        
        # Save metrics to disk
        self._save_metrics(run_id)
        
        logger.debug(f"Logged metric '{name}': {value} (run_id: {run_id})")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        run_id: Optional[str] = None,
        step: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
            run_id: Run identifier
            step: Training step or epoch
            timestamp: Timestamp for the metrics
        """
        timestamp = timestamp or datetime.datetime.now().isoformat()
        
        for name, value in metrics.items():
            self.log_metric(name, value, run_id, step, timestamp)
    
    def log_artifact(
        self,
        name: str,
        artifact: Any,
        artifact_type: str = "json",
        run_id: Optional[str] = None,
    ) -> str:
        """
        Log an artifact.
        
        Args:
            name: Artifact name
            artifact: Artifact data
            artifact_type: Type of artifact
            run_id: Run identifier
            
        Returns:
            Path to the saved artifact
        """
        run_id = run_id or "default"
        artifact_dir = self.artifacts_dir / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        if artifact_type == "json":
            artifact_path = artifact_dir / f"{name}.json"
            with open(artifact_path, "w") as f:
                json.dump(artifact, f, indent=4)
        
        elif artifact_type == "figure":
            artifact_path = artifact_dir / f"{name}.png"
            artifact.savefig(artifact_path)
            plt.close(artifact)
        
        elif artifact_type == "dataframe":
            artifact_path = artifact_dir / f"{name}.csv"
            artifact.to_csv(artifact_path, index=False)
        
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")
        
        logger.debug(f"Logged artifact '{name}' (run_id: {run_id})")
        
        return str(artifact_path)
    
    def get_metrics(
        self, 
        name: str, 
        run_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get metrics by name.
        
        Args:
            name: Metric name
            run_id: Run identifier
            
        Returns:
            List of metric values
        """
        run_id = run_id or "default"
        
        if run_id not in self._metrics or name not in self._metrics[run_id]:
            return []
        
        return self._metrics[run_id][name]
    
    def get_metrics_as_df(
        self, 
        name: str, 
        run_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get metrics as a pandas DataFrame.
        
        Args:
            name: Metric name
            run_id: Run identifier
            
        Returns:
            DataFrame with metric values
        """
        metrics = self.get_metrics(name, run_id)
        return pd.DataFrame(metrics)
    
    def compare_runs(
        self, 
        metric_name: str, 
        run_ids: List[str]
    ) -> pd.DataFrame:
        """
        Compare metric across multiple runs.
        
        Args:
            metric_name: Metric name to compare
            run_ids: List of run identifiers
            
        Returns:
            DataFrame with metric values for each run
        """
        result = {}
        
        for run_id in run_ids:
            metrics = self.get_metrics(metric_name, run_id)
            if metrics:
                # Get the most recent value for each run
                result[run_id] = metrics[-1]["value"]
        
        return pd.DataFrame({metric_name: result}).reset_index().rename(columns={"index": "run_id"})
    
    def plot_metric(
        self, 
        name: str, 
        run_id: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot a metric over time or steps.
        
        Args:
            name: Metric name
            run_id: Run identifier
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        metrics_df = self.get_metrics_as_df(name, run_id)
        
        if metrics_df.empty:
            raise ValueError(f"No metrics found for '{name}' (run_id: {run_id or 'default'})")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if "step" in metrics_df.columns:
            ax.plot(metrics_df["step"], metrics_df["value"])
            ax.set_xlabel("Step")
        else:
            timestamps = pd.to_datetime(metrics_df["timestamp"])
            ax.plot(timestamps, metrics_df["value"])
            ax.set_xlabel("Time")
        
        ax.set_ylabel(name)
        ax.set_title(f"{name} over time (run_id: {run_id or 'default'})")
        ax.grid(True)
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def _save_metrics(self, run_id: str) -> None:
        """Save metrics to disk."""
        if run_id in self._metrics:
            metrics_file = self.metrics_dir / f"{run_id}.json"
            with open(metrics_file, "w") as f:
                json.dump(self._metrics[run_id], f, indent=4)
    
    def load_metrics(self, run_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load metrics from disk.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dictionary of metrics
        """
        metrics_file = self.metrics_dir / f"{run_id}.json"
        
        if not metrics_file.exists():
            return {}
        
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        
        return metrics


# mlops_framework/versioning/manager.py
import os
import json
import shutil
import logging
import datetime
from