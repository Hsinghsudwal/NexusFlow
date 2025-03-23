import concurrent.futures
from typing import List, Dict, Optional
import time  # For simulating execution time of tasks


class Node:
    """Represents a node in the pipeline."""

    def __init__(self, name: str, dependencies: Optional[List['Node']] = None):
        """
        Initialize the node with a name and optional dependencies.

        Args:
            name (str): Name of the node.
            dependencies (List[Node], optional): List of nodes that should be executed before this one.
        """
        self.name = name
        self.dependencies = dependencies if dependencies is not None else []
        self.executed = False  # Track whether the node has been executed

    def execute(self, input_artifacts: Dict, config: Optional[Dict] = None):
        """Execute the node's task."""
        print(f"Executing node: {self.name}")
        # Simulate some execution time
        time.sleep(1)  # Replace with actual execution logic
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

    def node(
        func: Optional[Callable] = None,
        inputs: Optional[List[str]] = None,
        outputs: Optional[Union[str, List[str]]] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Union[Node, Callable]:
        """Decorator to create a node from a function."""
        
    # Handle case when decorator is used without parentheses
    if func is not None:
        return Node(
            func=func,
            inputs=inputs or [],
            outputs=[outputs] if isinstance(outputs, str) else (outputs or []),
            name=name or func.__name__,
            tags=tags or []
        )
    
    # Handle case when decorator is used with parentheses
    def decorator(function):
        return Node(
            func=function,
            inputs=inputs or [],
            outputs=[outputs] if isinstance(outputs, str) else (outputs or []),
            name=name or function.__name__,
            tags=tags or []
        )
    
    return decorator




# Configure logging
logging.basicConfig(level=logging.INFO)

# Define data processing nodes
@node(outputs="raw_data")
def load_data():
    """Load raw data from source."""
    # For example purposes, we're creating synthetic data
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

@node(inputs="raw_data", outputs=["X_data", "y_data"])
def preprocess_data(raw_data):
    """Extract features and target, handle missing values."""
    # Handle any missing values
    data_clean = raw_data.fillna(0)
    
    # Split into features and target
    X = data_clean.drop('churn', axis=1)
    y = data_clean['churn']
    
    return X, y

@node(inputs=["X_data", "y_data"], outputs=["X_train", "X_test", "y_train", "y_test"])
def split_data(X_data, y_data):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

@node(inputs=["X_train", "X_test"], outputs=["X_train_scaled", "X_test_scaled", "scaler"])
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

@node(inputs=["X_train_scaled", "y_train"], outputs="model")
def train_model(X_train_scaled, y_train):
    """Train a RandomForest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

@node(inputs=["model", "X_test_scaled", "y_test"], outputs=["predictions", "metrics"])
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

@node(inputs=["model", "X_data"], outputs="feature_importance")
def analyze_features(model, X_data):
    """Analyze feature importance."""
    feature_importance = pd.DataFrame({
        'feature': X_data.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance

# Create the preprocessing pipeline
preprocessing_pipeline = Pipeline(
    nodes=[
        load_data,
        preprocess_data,
        split_data,
        scale_features
    ],
    name="preprocessing_pipeline"
)

# Create the modeling pipeline
modeling_pipeline = Pipeline(
    nodes=[
        train_model,
        evaluate_model,
        analyze_features
    ],
    name="modeling_pipeline"
)

# Create the full pipeline
full_pipeline = Pipeline(
    nodes=[
        load_data,
        preprocess_data,
        split_data,
        scale_features,
        train_model,
        evaluate_model,
        analyze_features
    ],
    name="full_ml_pipeline"
)

if __name__ == "__main__":
    # Create project context
    context = ProjectContext("ml_project")
    
    # Run the full pipeline
    results = context.run_pipeline(full_pipeline)
    
    # Print results
    print(f"Model Accuracy: {results['metrics']['accuracy']:.4f}")
    print("\nFeature Importance:")
    print(results['feature_importance'])




# Example usage:

# Initialize nodes
# node1 = DataPreprocessingNode("Data Preprocessing")
# node2 = ModelTrainingNode("Model Training", dependencies=[node1])  # Node 2 depends on node1
# node3 = ModelEvaluationNode("Model Evaluation", dependencies=[node2])  # Node 3 depends on node2
# node4 = ModelDeploymentNode("Model Deployment", dependencies=[node3])  # Node 4 depends on node3

# Create pipeline and add nodes
# pipeline = Pipeline(name="MLOps Pipeline", description="End-to-End ML Pipeline", max_workers=2)
# pipeline.add_node(node1)
# pipeline.add_node(node2)
# pipeline.add_node(node3)
# pipeline.add_node(node4)

# Run the pipeline in parallel (ThreadPoolExecutor is used)
# pipeline.run(input_artifacts={}, config={})

# Create another pipeline without parallel execution (sequential)
# pipeline_sequential = Pipeline(name="Sequential MLOps Pipeline", description="Sequential ML Pipeline")
# pipeline_sequential.add_node(node1)
# pipeline_sequential.add_node(node2)
# pipeline_sequential.add_node(node3)
# pipeline_sequential.add_node(node4)

# Run the pipeline sequentially
# pipeline_sequential.run(input_artifacts={}, config={})

# Shutdown the executor (if using parallel mode)
# pipeline.shutdown()
