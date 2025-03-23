# src/monitoring/metrics.py
import time
import pandas as pd
import numpy as np
import json
import logging
import requests
from prometheus_client import Counter, Gauge, Histogram, push_to_gateway
from datetime import datetime, timedelta
import schedule
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """
    Model monitoring component to track performance metrics and data drift.
    """
    
    def __init__(self, config):
        """
        Initialize the model monitor.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.model_name = config.get('model_name', 'churn_predictor')
        self.service_url = config.get('model_service_url')
        self.pushgateway_url = config.get('prometheus_pushgateway_url')
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        # Load reference data if available
        self.reference_data = None
        reference_data_path = config.get('reference_data_path')
        if reference_data_path and os.path.exists(reference_data_path):
            self.reference_data = pd.read_parquet(reference_data_path)
            logger.info(f"Loaded reference data from {reference_data_path}")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Model performance metrics
        self.prediction_counter = Counter(
            'model_prediction_count', 
            'Number of predictions',
            ['model', 'version']
        )
        
        self.latency_histogram = Histogram(
            'model_prediction_latency', 
            'Prediction latency in seconds',
            ['model', 'version'],
            buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
        )
        
        self.error_counter = Counter(
            'model_prediction_error_count', 
            'Number of prediction errors',
            ['model', 'version', 'error_type']
        )
        
        # Data drift metrics
        self.feature_drift_gauge = Gauge(
            'model_feature_drift', 
            'Feature drift score',
            ['model', 'version', 'feature']
        )
        
        self.performance_gauge = Gauge(
            'model_performance_metric', 
            'Model performance metric',
            ['model', 'version', 'metric']
        )
    
    def record_prediction(self, prediction_input, prediction_output, prediction_time):
        """
        Record a single prediction.
        
        Args:
            prediction_input (dict): Input data for prediction
            prediction_output (dict): Model prediction result
            prediction_time (float): Time taken for prediction
        """
        model_version = self.config.get('model_version', 'latest')
        
        # Increment prediction counter
        self.prediction_counter.labels(
            model=self.model_name,
            version=model_version
        ).inc()
        
        # Record prediction latency
        self.latency_histogram.labels(
            model=self.model_name,
            version=model_version
        ).observe(prediction_time)
        
        # Push metrics to Prometheus Pushgateway
        if self.pushgateway_url:
            try:
                push_to_gateway(
                    self.pushgateway_url,
                    job=f"{self.model_name}_metrics",
                    registry=None
                )
            except Exception as e:
                logger.error(f"Failed to push metrics to Pushgateway: {str(e)}")
    
    def record_error(self, error_type):
        """
        Record a prediction error.
        
        Args:
            error_type (str): Type of error that occurred
        """
        model_version = self.config.get('model_version', 'latest')
        
        # Increment error counter
        self.error_counter.labels(
            model=self.model_name,
            version=model_version,
            error_type=error_type
        ).inc()
        
        # Push metrics to Prometheus Pushgateway
        if self.pushgateway_url:
            try:
                push_to_gateway(
                    self.pushgateway_url,
                    job=f"{self.model_name}_errors",
                    registry=None
                )
            except Exception as e:
                logger.error(f"Failed to push error metrics