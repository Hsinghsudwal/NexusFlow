# monitoring.py - Model Monitoring
import pandas as pd
import numpy as np
from scipy import stats
import json
import time
from datetime import datetime
import threading
from config import MLOpsConfig

class ModelMonitor:
    def __init__(self, model_name, model_version, config=None):
        self.config = config or MLOpsConfig()
        self.model_name = model_name
        self.model_version = model_version
        
        # Initialize storage for prediction data
        self.predictions = []
        self.baseline_data = None
        self.drift_thresholds = {}
        self.alerts = []
        
        # Set up monitoring thread
        self.monitoring_active = False
    
    def load_baseline(self, baseline_path):
        """Load baseline data for drift detection"""
        self.baseline_data = pd.read_parquet(baseline_path)
        
        # Calculate statistical properties for each feature
        for column in self.baseline_data.columns:
            if pd.api.types.is_numeric_dtype(self.baseline_data[column]):
                self.drift_thresholds[column] = {
                    "mean": self.baseline_data[column].mean(),
                    "std": self.baseline_data[column].std(),
                    "p05": self.baseline_data[column].quantile(0.05),
                    "p95": self.baseline_data[column].quantile(0.95),
                    "threshold": 0.05  # p-value threshold for KS test
                }
    
    def add_prediction(self, input_data, prediction, timestamp=None):
        """Add a prediction for monitoring"""
        timestamp = timestamp or datetime.now().isoformat()
        
        record = {
            "timestamp": timestamp,
            "input": input_data,
            "prediction": prediction
        }
        
        self.predictions.append(record)
        
        # Check for drift if we have enough data
        if len(self.predictions) % 100 == 0:
            self.check_drift()
    
    def check_drift(self):
        """Check for data and concept drift"""
        if not self.baseline_data is not None or len(self.predictions) < 50:
            return
        
        # Extract input data from predictions
        recent_inputs = []
        for record in self.predictions[-100:]:
            if isinstance(record["input"], dict):
                recent_inputs.append(record["input"])
        
        if not recent_inputs:
            return
            
        # Convert to DataFrame
        recent_df = pd.DataFrame(recent_inputs)
        
        # Check each numeric feature for drift
        drift_detected = False
        drift_report = {"timestamp": datetime.now().isoformat(), "features": {}}
        
        for column in self.drift_thresholds:
            if column in recent_df.columns and pd.api.types.is_numeric_dtype(recent_df[column]):
                # Get baseline and recent data
                baseline_values = self.baseline_data[column].dropna()
                recent_values = recent_df[column].dropna()
                
                if len(recent_values) < 10:
                    continue
                    
                # Basic statistics comparison
                recent_mean = recent_values.mean()
                recent_std = recent_values.std()
                baseline_mean = self.drift_thresholds[column]["mean"]
                baseline_std = self.drift_thresholds[column]["std"]
                
                # Calculate z-score for mean shift
                mean_z_score = abs(recent_mean - baseline_mean) / baseline_std
                
                # Perform KS test
                ks_statistic, p_value = stats.ks_2samp(baseline_values, recent_values)
                
                # Check if drift is detected
                threshold = self.drift_thresholds[column]["threshold"]
                if p_value < threshold or mean_z_score > 3:
                    drift_detected = True
                    drift_report["features"][column] = {
                        "drift_detected": True,
                        "p_value": p_value,
                        "ks_statistic": ks_statistic,
                        "mean_shift": {
                            "baseline": baseline_mean,
                            "recent": recent_mean,
                            "z_score": mean_z_score
                        }
                    }
        
        if drift_detected:
            drift_report["status"] = "DRIFT_DETECTED"
            self.alerts.append(drift_report)
            self._send_alert(drift_report)
        
        return drift_report
    
    def _send_alert(self, alert_data):
        """Send an alert (would be implemented with a notification system)"""
        print(f"⚠️ ALERT: Data drift detected for model {self.model_name} v{self.model_version}")
        print(json.dumps(alert_data, indent=2))
        
        # In a real implementation, this would send to an alerting system
        # For example: email, Slack, PagerDuty, etc.
    
    def start_monitoring(self, interval_seconds=300):
        """Start background monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self.check_drift()
                    time.sleep(interval_seconds)
                except Exception as e:
                    print(f"Error in monitoring loop: {e}")
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        print(f"Started monitoring for model {self.model_name} v{self.model_version}")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        print(f"Stopped monitoring for model {self.model_name} v{self.model_version}")