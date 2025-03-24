# feature_store.py - Feature store setup and management

import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

from feast import (
    Entity, Feature, FeatureView, FileSource, 
    ValueType, FeatureStore, RepoConfig, 
    OnDemandFeatureView, RequestSource
)
from feast.infra.offline_stores.file_source import FileSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Int64, String
import feast.field as field

# Define the feature repository
REPO_PATH = os.getenv("FEATURE_STORE_PATH", "feature_repo")

def init_feature_repo(repo_path: str = REPO_PATH):
    """Initialize feature repository directory structure."""
    os.makedirs(repo_path, exist_ok=True)
    
    # Create feature_store.yaml
    feature_store_config = {
        "project": "ml_project",
        "registry": f"{repo_path}/registry.db",
        "provider": "local",
        "online_store": {
            "type": "redis",
            "connection_string": "redis://redis:6379/0"
        },
        "offline_store": {
            "type": "file"
        },
        "entity_key_serialization_version": 2
    }
    
    with open(f"{repo_path}/feature_store.yaml", "w") as f:
        yaml.dump(feature_store_config, f)
    
    print(f"Initialized feature repository at {repo_path}")

def generate_example_data(output_path: str):
    """Generate example data for feature store."""
    # Create customer entities
    num_customers = 1000
    customer_ids = [f"C{i:04d}" for i in range(num_customers)]
    
    # Transaction history
    now = datetime.now()
    transactions = []
    
    for customer_id in customer_ids:
        # Generate random number of transactions
        num_transactions = np.random.randint(1, 20)
        
        for _ in range(num_transactions):
            # Random transaction in the last 90 days
            timestamp = now - timedelta(days=np.random.randint(1, 90))
            amount = np.random.randint(100, 10000) / 100
            category = np.random.choice(["grocery", "retail", "travel", "dining", "other"])
            
            transactions.append({
                "customer_id": customer_id,
                "timestamp": timestamp,
                "amount": amount,
                "category": category
            })
    
    # Convert to DataFrame and save
    transactions_df = pd.DataFrame(transactions)
    transactions_df["event_timestamp"] = transactions_df["timestamp"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transactions_df.to_parquet(output_path)
    
    print(f"Generated {len(transactions_df)} transaction records for {num_customers} customers")
    return transactions_df

def create_transaction_features():
    """Create feature definitions for transaction data."""
    # Entity definition
    customer = Entity(
        name="customer",
        value_type=ValueType.STRING,
        description="Customer identifier"
    )
    
    # Data source
    transaction_source = FileSource(
        path="data/transactions.parquet",
        event_timestamp_column="event_timestamp",
        created_timestamp_column=None,
    )
    
    # Feature view
    transaction_stats = FeatureView(
        name="transaction_stats",
        entities=["customer"],
        ttl=timedelta(days=365),
        schema=[
            field.Field(name="amount", dtype=Float32),
            field.Field(name="category", dtype=String),
        ],
        online=True,
        source=transaction_source,
        tags={"team": "ml_team"},
    )
    
    # On-demand feature view for derived features
    request_schema = [
        field.Field(name="transaction_amount", dtype=Float32),
    ]
    
    @on_demand_feature_view(
        sources=[
            transaction_stats,
            RequestSource(schema=request_schema)
        ],
        schema=[
            field.Field(name="transaction_amount_scaled", dtype=Float32),
            field.Field(name="is_large_transaction", dtype=Int64),
        ]
    )
    def transaction_transformations(inputs: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame()
        
        # Apply scaling to transaction amount
        avg_amount = inputs["transaction_stats__amount"].mean()
        df["transaction_amount_scaled"] = inputs["transaction_amount"] / avg_amount
        
        # Flag large transactions
        df["is_large_transaction"] = (inputs["transaction_amount"] > 100).astype(int)
        
        return df
    
    # Return all feature definitions
    return [customer, transaction_stats, transaction_transformations]

def save_feature_definitions(repo_path: str, features: List):
    """Save feature definitions to the repository."""
    # Create features.py in the repository
    feature_code = """
import os
from datetime import timedelta
import pandas as pd

from feast import (
    Entity, Feature, FeatureView, FileSource, 
    ValueType, FeatureStore, RepoConfig
)
from feast.infra.offline_stores.file_source import FileSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.request_source import RequestSource
from feast.types import Float32, Int64, String
import feast.field as field

# Entity definition
customer = Entity(
    name="customer",
    value_type=ValueType.STRING,
    description="Customer identifier"
)

# Data source
transaction_source = FileSource(
    path="data/transactions.parquet",
    event_timestamp_column="event_timestamp"
)

# Feature view
transaction_stats = FeatureView(
    name="transaction_stats",
    entities=["customer"],
    ttl=timedelta(days=365),
    schema=[
        field.Field(name="amount", dtype=Float32),
        field.Field(name="category", dtype=String),
    ],
    online=True,
    source=transaction_source,
    tags={"team": "ml_team"},
)

# On-demand feature view for derived features
request_schema = [
    field.Field(name="transaction_amount", dtype=Float32),
]

@on_demand_feature_view(
    sources=[
        transaction_stats,
        RequestSource(schema=request_schema)
    ],
    schema=[
        field.Field(name="transaction_amount_scaled", dtype=Float32),
        field.Field(name="is_large_transaction", dtype=Int64),
    ]
)
def transaction_transformations(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    
    # Apply scaling to transaction amount
    avg_amount = inputs["transaction_stats__amount"].mean()
    df["transaction_amount_scaled"] = inputs["transaction_amount"] / avg_amount
    
    # Flag large transactions
    df["is_large_transaction"] = (inputs["transaction_amount"] > 100).astype(int)
    
    return df
"""

    with open(f"{repo_path}/features.py", "w") as f:
        f.write(feature_code)
    
    print(f"Saved feature definitions to {repo_path}/features.py")

def deploy_feature_store(repo_path: str):
    """Deploy feature store and register features."""
    # Initialize store
    store = FeatureStore(repo_path=repo_path)
    
    # Apply feature definitions
    store.apply([])  # Features are imported from features.py
    
    # Materialize features to online store for real-time serving
    store.materialize_incremental(end_date=datetime.now())
    
    print("Feature store deployed and features registered")
    return store

def sample_feature_retrieval(store: FeatureStore):
    """Sample code to retrieve features."""
    # Get historical features for training
    entity_df = pd.DataFrame(
        {
            "customer_id": ["C0001", "C0002", "C0003"],
            "event_timestamp": [
                datetime.now(),
                datetime.now(),
                datetime.now()
            ]
        }
    )
    
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "transaction_stats:amount",
            "transaction_stats:category"
        ],
    ).to_df()
    
    print("Historical features for training:")
    print(training_df.head())
    
    # Get online features for prediction
    online_features = store.get_online_features(
        features=[
            "transaction_stats:amount",
            "transaction_stats:category"
        ],
        entity_rows=[
            {"customer_id": "C0001"}
        ]
    ).to_dict()
    
    print("Online features for prediction:")
    print(online_features)

if __name__ == "__main__":
    # Create feature repository
    init_feature_repo()
    
    # Generate example data
    generate_example_data("data/transactions.parquet")
    
    # Create and save feature definitions
    features = create_transaction_features()
    save_feature_definitions(REPO_PATH, features)
    
    # Deploy feature store
    store = deploy_feature_store(REPO_PATH)
    
    # Sample feature retrieval
    sample_feature_retrieval(store)
