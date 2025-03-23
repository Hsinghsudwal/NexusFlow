# Example: Creating a data pipeline with all the enhanced features
from enhanced_pipeline_framework import (
    Pipeline, create_node, create_condition, create_validator,
    ExecutionMode, RetryStrategy, NodeStatus
)
import time

# Define node functions
def fetch_data(input_data=None):
    print("Fetching data...")
    return [1, 2, 3, 4, 5]

def validate_numbers(data):
    # Example validator function - returns (is_valid, error_message)
    if not isinstance(data, list):
        return False, "Data must be a list"
    if not all(isinstance(x, (int, float)) for x in data):
        return False, "All items must be numbers"
    return True, ""

def process_data(data):
    print(f"Processing data: {data}")
    # Simulating occasional failures
    if sum(data) % 7 == 0:
        raise ValueError("Simulated random error")
    return [x * 2 for x in data]

def analyze_data(data):
    print(f"Analyzing data: {data}")
    return {"sum": sum(data), "avg": sum(data)/len(data), "count": len(data)}

def format_results(data):
    print(f"Formatting results: {data}")
    return {"result": data, "timestamp": time.time()}

def save_results(data):
    print(f"Saving results: {data}")
    return f"Saved {data['result']['count']} items"

def notify_users(data):
    print(f"Sending notification about: {data}")
    return "Notification sent"

# Create pipeline
pipeline = Pipeline(
    name="Enhanced Data Processing Pipeline",
    description="Fetches, processes, formats and saves data with error handling", 
    execution_mode=ExecutionMode.SEQUENTIAL
)

# Create validators
number_validator = create_validator(validate_numbers, "Validate input is a list of numbers")

# Create nodes with error handling and validation
node1 = create_node(fetch_data, "Fetch Data", "Fetches raw data", "node_1")
node2 = create_node(process_data, "Process Data", "Doubles each value", "node_2")
node2.add_validator(number_validator)
node2.set_retry_config(
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    max_attempts=3,
    delay=1.0
)

node3a = create_node(analyze_data, "Analyze Data", "Calculates statistics", "node_3a")
node3b = create_node(format_results, "Format Results", "Formats for storage", "node_3b")
node4 = create_node(save_results, "Save Results", "Saves to database", "node_4")
node5 = create_node(notify_users, "Notify Users", "Sends notifications", "node_5")

# Add nodes to pipeline
pipeline.add_node(node1)
pipeline.add_node(node2)
pipeline.add_node(node3a)
pipeline.add_node(node3b)
pipeline.add_node(node4)
pipeline.add_node(node5)

# Link nodes with regular and conditional branches
node1.add_next(node2)

# Create a condition for large datasets
large_dataset_condition = create_condition(
    lambda data: len(data) > 3, 
    "Large dataset branch"
)

# Add conditional branching
node2.add_next(node3a)
node2.add_conditional_next(node3b, large_dataset_condition)
node3a.add_next(node4)
node3b.add_next(node4)
node4.add_next(node5)

# Set start node
pipeline.set_start_node(node1)

# Run pipeline
try:
    results = pipeline.run()
    print("Pipeline executed successfully")
    print(results)
    
    # Get execution statistics
    stats = pipeline.get_execution_stats()
    print(stats)
    
    # Save pipeline state for later analysis
    state_file = pipeline.save_state()
    print(f"Pipeline state saved to {state_file}")
    
    # Generate visualization
    vis_file = pipeline.visualize(view=True)
    print(f"Pipeline visualization saved to {vis_file}")
    
except Exception as e:
    print(f"Pipeline execution failed: {e}")