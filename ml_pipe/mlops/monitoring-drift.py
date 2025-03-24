---
# monitoring/drift_detection.py - Feature drift detection script

import pandas as pd
import numpy as np
import redis
import json
import time
import os
import logging
from datetime import datetime
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection for feature store
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

# Reference dataset location
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "/data/reference_data.parquet")

def load_reference_data():
    """Load reference dataset used for drift detection."""
    try:
        return pd.read_parquet(REFERENCE_DATA_PATH)
    except Exception as e:
        logger.error(f"Error loading reference data: {e}")
        return None

def calculate_reference_statistics(df):
    """Calculate reference statistics for numerical features."""
    stats_dict = {}
    
    for column in df.select_dtypes(include=[np.number]).columns:
        column_stats = {
            "mean": float(df[column].mean()),
            "std": float(df[column].std()),
            "min": float(df[column].min()),
            "max": float(df[column].max()),
            "q25": float(df[column].quantile(0.25)),
            "q50": float(df[column].quantile(0.50)),
            "q75": float(df[column].quantile(0.75))
        }
        stats_dict[column] = column_stats
        
        # Store in Redis
        redis_client.set(f"feature_stats:{column}", json.dumps(column_stats))
        logger.info(f"Calculated statistics for feature {column}")
    
    return stats_dict

def get_recent_predictions(n=1000):
    """Get recent prediction data from logs."""
    logs = []
    for i in range(n):
        log_entry = redis_client.lindex("prediction_logs", i)
        if log_entry is None:
            break
        logs.append(json.loads(log_entry))
    
    if not logs:
        return None
    
    # Convert to DataFrame
    features_list = [log["features"] for log in logs]
    df = pd.DataFrame(features_list)
    return df

def detect_drift(reference_stats, current_data):
    """Detect drift in current data compared to reference."""
    drift_detected = False
    drift_report = {}
    
    for column in current_data.select_dtypes(include=[np.number]).columns:
        if column not in reference_stats:
            continue
            
        # Basic statistical tests
        ref_stats = reference_stats[column]
        current_values = current_data[column].dropna()
        
        if len(current_values) < 30:
            continue
            
        # Mean shift detection
        mean_diff_sigma = abs(current_values.mean() - ref_stats["mean"]) / ref_stats["std"]
        
        # KS-test for distribution shift
        try:
            reference_sample = np.random.normal(
                ref_stats["mean"], 
                ref_stats["std"], 
                min(1000, len(current_values))
            )
            ks_stat, ks_pvalue = stats.ks_2samp(reference_sample, current_values)
        except Exception:
            ks_stat, ks_pvalue = 0, 1
        
        # Check for drift
        is_drift = (mean_diff_sigma > 3) or (ks_pvalue < 0.01 and ks_stat > 0.1)
        
        if is_drift:
            drift_detected = True
            drift_report[column] = {
                "mean_diff_sigma": float(mean_diff_sigma),
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "current_mean": float(current_values.mean()),
                "reference_mean": ref_stats["mean"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Log drift event
            logger.warning(f"Drift detected in feature {column}: {drift_report[column]}")
    
    # Store drift report
    if drift_detected:
        report_key = f"drift_report:{int(time.time())}"
        redis_client.set(report_key, json.dumps(drift_report))
        redis_client.expire(report_key, 60*60*24*7)  # Keep for 7 days
    
    return drift_detected, drift_report

def run_drift_detection():
    """Main function to run drift detection process."""
    logger.info("Starting drift detection")
    
    # Load reference data
    reference_data = load_reference_data()
    if reference_data is None:
        logger.error("Cannot run drift detection without reference data")
        return
    
    # Calculate reference statistics
    reference_stats = calculate_reference_statistics(reference_data)
    
    # Get recent prediction data
    current_data = get_recent_predictions()
    if current_data is None or current_data.empty:
        logger.info("No recent prediction data available for drift detection")
        return
    
    # Detect drift
    drift_detected, drift_report = detect_drift(reference_stats, current_data)
    
    if drift_detected:
        logger.warning(f"Drift detected in {len(drift_report)} features")
        # In a real system, this would trigger retraining or alerts
    else:
        logger.info("No significant drift detected")

if __name__ == "__main__":
    run_drift_detection()
