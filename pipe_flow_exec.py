import concurrent.futures
from typing import List, Dict, Optional, Callable, Union
import time  # For simulating execution time of tasks
import logging



class Node:
    """Represents a node in the pipeline. Can also be used as a decorator."""

    def __init__(self, func: Optional[Callable] = None, inputs: Optional[List[str]] = None,
                 outputs: Optional[Union[str, List[str]]] = None, name: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        """
        Initialize the node with a function and its attributes.
        
        Args:
            func (Callable, optional): The function to be wrapped by this node.
            inputs (List[str], optional): List of input artifacts required by the node.
            outputs (Union[str, List[str]], optional): List or single output artifact produced by the node.
            name (str, optional): Name of the node.
            tags (List[str], optional): Tags associated with the node.
        """
        if func is not None:
            self.func = func
            self.name = name or func.__name__
            self.inputs = inputs or []
            self.outputs = [outputs] if isinstance(outputs, str) else (outputs or [])
            self.tags = tags or []
            self.executed = False  # Track whether the node has been executed
        else:
            self.func = None
            self.name = name
            self.inputs = inputs or []
            self.outputs = outputs or []
            self.tags = tags or []

    def __call__(self, *args, **kwargs):
        """Allow the Node class to be used as a decorator."""
        return Node(func=self.func, inputs=self.inputs, outputs=self.outputs, name=self.name, tags=self.tags)

    def execute(self, input_artifacts: Dict, config: Optional[Dict] = None):
        """Execute the node's task."""
        print(f"Executing node: {self.name}")
        time.sleep(1)  # Simulate execution time, replace with actual logic
        self.executed = True
        print(f"Node {self.name} execution complete.")




class Pipeline:
    def __init__(self, name: str, description: str, max_workers: Optional[int] = None):
        """
        Initialize the pipeline.

        Args:
            name (str): Name of the pipeline.
            description (str): Description of the pipeline.
            max_workers (int, optional): Maximum number of workers for parallel execution.
        """
        self.name = name
        self.description = description
        self.nodes: List[Node] = []  # List of nodes in the pipeline
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) if max_workers else None

    def __call__(self, pipeline_class):
        """This method allows the Pipeline to be used as a decorator."""
        # Add any logic to populate the pipeline's nodes based on the decorated class
        pipeline_class.pipeline = self
        return pipeline_class

    def add_node(self, node: Node):
        """Add a node to the pipeline."""
        self.nodes.append(node)

    def run(self, input_artifacts: Dict, config: Optional[Dict] = None):
        """Run the pipeline based on whether we have an executor or not."""
        print(f"Running pipeline '{self.name}'...")

        if self.executor:
            self._run_in_parallel(input_artifacts, config)
        else:
            self._run_sequentially(input_artifacts, config)

    def _run_sequentially(self, input_artifacts: Dict, config: Optional[Dict] = None):
        """Run nodes one after another (sequential execution)."""
        for node in self.nodes:
            # Ensure all dependencies are executed before running the current node
            self._execute_node_with_dependencies(node, input_artifacts, config)
            node.execute(input_artifacts, config)

    def _run_in_parallel(self, input_artifacts: Dict, config: Optional[Dict] = None):
        """Run nodes concurrently using ThreadPoolExecutor (parallel execution)."""
        futures: List[concurrent.futures.Future] = []

        for node in self.nodes:
            # Ensure all dependencies are executed before running the current node
            self._execute_node_with_dependencies(node, input_artifacts, config)

            # Submit each node for parallel execution
            future = self.executor.submit(node.execute, input_artifacts, config)
            futures.append(future)

        # Wait for all tasks to complete
        for future in futures:
            future.result()  # This will block until the task is completed

    def _execute_node_with_dependencies(self, node: Node, input_artifacts: Dict, config: Optional[Dict] = None):
        """Ensure that all dependencies of a node are executed before executing the node itself."""
        for dependency in node.dependencies:
            if not dependency.executed:
                print(f"Executing dependency: {dependency.name}")
                dependency.execute(input_artifacts, config)

    def shutdown(self):
        """Shutdown the executor after pipeline execution."""
        if self.executor:
            self.executor.shutdown()

  


# Configure logging
logging.basicConfig(level=logging.INFO)

# Define data processing nodes
@Node(outputs="raw_data")
def load_data():
    """Load raw data from source."""
    np.random.seed(42)
    n_samples = 1000

    # Create features (age, income, education_level, etc.)
    X = np.random.randn(n_samples, 5)

    # Create target (customer churn: 0 or 1)
    y = (X[:, 0] + X[:, 2] > 0).astype(int)

    # Create dataframe
    df = pd.DataFrame(
        X, 
        columns=['age', 'income', 'education', 'tenure', 'products_owned']
    )
    df['churn'] = y

    return df

@Node(inputs="raw_data", outputs=["X_data", "y_data"])
def preprocess_data(raw_data):
    """Extract features and target, handle missing values."""
    data_clean = raw_data.fillna(0)
    X = data_clean.drop('churn', axis=1)
    y = data_clean['churn']
    return X, y

@Node(inputs=["X_data", "y_data"], outputs=["X_train", "X_test", "y_train", "y_test"])
def split_data(X_data, y_data):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@Node(inputs=["X_train", "X_test"], outputs=["X_train_scaled", "X_test_scaled", "scaler"])
def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    return X_train_scaled, X_test_scaled, scaler

@Node(inputs=["X_train_scaled", "y_train"], outputs="model")
def train_model(X_train_scaled, y_train):
    """Train a RandomForest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

@Node(inputs=["model", "X_test_scaled", "y_test"], outputs=["predictions", "metrics"])
def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate model performance."""
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': report
    }
    
    return predictions, metrics

@Node(inputs=["model", "X_data"], outputs="feature_importance")
def analyze_features(model, X_data):
    """Analyze feature importance."""
    feature_importance = pd.DataFrame({
        'feature': X_data.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance

# Create the preprocessing pipeline
preprocessing_pipeline = Pipeline(
    name="preprocessing_pipeline",
    description="Pipeline for data preprocessing"
)

preprocessing_pipeline.add_node(load_data)
preprocessing_pipeline.add_node(preprocess_data)
preprocessing_pipeline.add_node(split_data)
preprocessing_pipeline.add_node(scale_features)

# Create the modeling pipeline
modeling_pipeline = Pipeline(
    name="modeling_pipeline",
    description="Pipeline for model training and evaluation"
)

modeling_pipeline.add_node(train_model)
modeling_pipeline.add_node(evaluate_model)
modeling_pipeline.add_node(analyze_features)

# Create the full pipeline
full_pipeline = Pipeline(
    name="full_ml_pipeline",
    description="Complete ML pipeline"
)

full_pipeline.add_node(load_data)
full_pipeline.add_node(preprocess_data)
full_pipeline.add_node(split_data)
full_pipeline.add_node(scale_features)
full_pipeline.add_node(train_model)
full_pipeline.add_node(evaluate_model)
full_pipeline.add_node(analyze_features)

if __name__ == "__main__":
    # Run the full pipeline
    full_pipeline.run(input_artifacts={}, config={})

    # Example: shutdown the executor after running the pipeline
    full_pipeline.shutdown()

# Define the pipeline using the decorator
@Pipeline(name="full_ml_pipeline", description="Complete ML pipeline")
class FullPipeline:
    pass

  or

# Create and run the pipeline
if __name__ == "__main__":
    # Create the full pipeline instance
    pipeline_instance = FullPipeline()


    input_artifacts = {"input_data": "some_data"}
    config = {"some_config": "value"}

    pipeline.pipeline.run(input_artifacts, config)
    # Run the pipeline
    pipeline_instance.run_pipeline(input_artifacts={}, config={})

    # Shutdown the executor after running the pipeline
    pipeline_instance.shutdown()
