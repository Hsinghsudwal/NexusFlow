import concurrent.futures
import uuid
from typing import List, Dict, Optional, Callable, Union
import time  # For simulating execution time of tasks
import logging
from datetime import datetime
import networkx as nx  # Importing networkx for DAG representation

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)





class Experiment:
    def __init__(self, pipeline_id: Optional[str] = None, pipeline_name: Optional[str] = None, 
                 start_time: Optional[datetime] = None, description: Optional[str] = None):
        """
        Initialize an experiment for tracking pipeline execution.

        Args:
            pipeline_id (str, optional): The unique ID of the pipeline (if not provided, will generate a new one).
            pipeline_name (str, optional): The name of the pipeline.
            start_time (datetime, optional): The start time of the experiment (current time if not provided).
            description (str, optional): Optional description for the experiment.
        """
        self.pipeline_id = pipeline_id
        self.pipeline_name = pipeline_name
        self.start_time = start_time
        self.end_time = None
        self.status = "running"
        self.description = description
        self.metadata = {}
        
    def update_status(self, status: str):
        """Update the status of the experiment."""
        self.status = status

    def set_end_time(self, end_time: datetime):
        """Set the end time of the experiment."""
        self.end_time = end_time

    def save(self):
        """Save the experiment details to a JSON file (could also be a database)."""
        experiment_data = {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "description": self.description,
            "metadata": self.metadata
        }
        # Save to a file (for simplicity here, you can change this to a database or other storage)
        with open(f"experiment_{self.pipeline_id}.json", "w") as f:
            json.dump(experiment_data, f, indent=4)

    def add_metadata(self, key: str, value: str):
        """Add metadata to the experiment."""
        self.metadata[key] = value




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

        duration = datetime.now() - start_time
        logger.info(f"Node {self.name} completed in {duration.total_seconds():.2f}s")




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
        #  self.steps = {}  # For the second pipeline (Step-based)
        self.dag = nx.DiGraph()  # For the directed acyclic graph representation
        self.id = str(uuid.uuid4())  # Unique pipeline ID for tracking

    def __call__(self, pipeline_class):
        """This method allows the Pipeline to be used as a decorator."""
        # Add any logic to populate the pipeline's nodes based on the decorated class
        pipeline_class.pipeline = self
        return pipeline_class

    def add_node(self, node: Node):
        """Add a node to the pipeline."""
        self.nodes.append(node)
        self.dag.add_node(node.name)

        # Add dependencies to the DAG
        if dependencies:
            for dep in dependencies:
                self.dag.add_edge(dep.name, node.name)

        # Verify that the DAG remains acyclic
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("Pipeline contains a cycle in the node dependencies.")


    # def run(self, input_artifacts: Dict, config: Optional[Dict] = None):
    #     """Run the pipeline based on whether we have an executor or not."""
    #     print(f"Running pipeline '{self.name}'...")

        # Use topological sorting to respect dependencies
        # sorted_nodes = list(nx.topological_sort(self.dag))

        # if self.executor:
        #     self._run_in_parallel(input_artifacts, config)
        # else:
        #     self._run_sequentially(input_artifacts, config)

    def run(self, input_artifacts: Dict, config: Optional[Dict] = None):
        """Run the pipeline and create an experiment to track the execution."""
        logger.info(f"Running pipeline '{self.name}'...")

        # Create the experiment object
        experiment = Experiment(
            pipeline_id=self.id,
            pipeline_name=self.name,
            start_time=datetime.now(),
            description=self.description
        )

        # Use topological sorting to respect dependencies
        sorted_nodes = list(nx.topological_sort(self.dag))

        try:
            if self.executor:
                self._run_in_parallel(sorted_nodes, input_artifacts, config)
            else:
                self._run_sequentially(sorted_nodes, input_artifacts, config)

            # Mark experiment as completed
            experiment.update_status("completed")
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            experiment.update_status("failed")

        # Set the end time and save the experiment
        experiment.set_end_time(datetime.now())
        experiment.save()


    # def _run_sequentially(self, input_artifacts: Dict, config: Optional[Dict] = None):
    #     """Run nodes one after another (sequential execution)."""
    #     for node in self.nodes:
    #         # Ensure all dependencies are executed before running the current node
    #         self._execute_node_with_dependencies(node, input_artifacts, config)
    #         node.execute(input_artifacts, config)

    # def _run_in_parallel(self, input_artifacts: Dict, config: Optional[Dict] = None):
    #     """Run nodes concurrently using ThreadPoolExecutor (parallel execution)."""
    #     futures: List[concurrent.futures.Future] = []
        # for node in self.nodes:
        #     # Ensure all dependencies are executed before running the current node
        #     self._execute_node_with_dependencies(node, input_artifacts, config)

        #     # Submit each node for parallel execution
        #     future = self.executor.submit(node.execute, input_artifacts, config)
        #     futures.append(future)

        # # Wait for all tasks to complete
        # for future in futures:
        #     future.result() 


    def _run_sequentially(self, sorted_nodes: List[str], input_artifacts: Dict, config: Optional[Dict] = None):
        """Run nodes one after another (sequential execution)."""
        for node_name in sorted_nodes:
            node = next(n for n in self.nodes if n.name == node_name)
            logger.info(f"Executing node '{node.name}' sequentially...")
            self._execute_node_with_dependencies(node, input_artifacts, config)
            node.execute(input_artifacts, config)


    def _run_in_parallel(self, sorted_nodes: List[str], input_artifacts: Dict, config: Optional[Dict] = None):
        """Run nodes concurrently using ThreadPoolExecutor (parallel execution)."""
        futures = []

        for node_name in sorted_nodes:
            node = next(n for n in self.nodes if n.name == node_name)
            logger.info(f"Executing node '{node.name}' in parallel...")
            self._execute_node_with_dependencies(node, input_artifacts, config)

            # Submit node execution for parallel execution
            future = self.executor.submit(node.execute, input_artifacts, config)
            futures.append(future)

        # Wait for all tasks to complete
        for future in futures:
            future.result()  # This will block until the task is completed

    # def _execute_node_with_dependencies(self, node: Node, input_artifacts: Dict, config: Optional[Dict] = None):
    #     """Ensure that all dependencies of a node are executed before executing the node itself."""
    #     for dependency in node.dependencies:
    #         if not dependency.executed:
    #             print(f"Executing dependency: {dependency.name}")
    #             dependency.execute(input_artifacts, config)


    def _execute_node_with_dependencies(self, node: Node, input_artifacts: Dict, config: Optional[Dict] = None):
        """Ensure that all dependencies of a node are executed before running the node itself."""
        for dep_name in self.dag.predecessors(node.name):
            dep_node = next(n for n in self.nodes if n.name == dep_name)
            if not dep_node.executed:
                logger.info(f"Executing dependency: {dep_node.name}")
                dep_node.execute(input_artifacts, config)


    def shutdown(self):
        """Shutdown the executor after pipeline execution."""
        if self.executor:
            self.executor.shutdown()

    
    # def run_step_based(self, config=None):
    #     """Execute the pipeline steps based on the DAG (topological sorting)."""
    #     logger.info(f"Running step-based pipeline '{self.name}'...")

    #     # Create an experiment (assuming you have an `Experiment` class)
    #     experiment = Experiment(pipeline_id=self.id, pipeline_name=self.name, start_time=datetime.now())

    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = {}

    #         # Execute steps based on topological order (dependencies are respected)
    #         for node_name in nx.topological_sort(self.dag):
    #             node = next(n for n in self.nodes if n.name == node_name)
    #             input_artifacts = {}

    #             # Gather the artifacts from all predecessors (dependencies)
    #             for dep_name in self.dag.predecessors(node_name):
    #                 dep_node = next(n for n in self.nodes if n.name == dep_name)
    #                 for key, artifact in dep_node.artifacts.items():
    #                     input_artifacts[key] = artifact.value

    #             # Submit each node for parallel execution
    #             futures[node_name] = executor.submit(node.execute, input_artifacts, config)

    #         # Wait for all futures to complete
    #         for future in futures.values():
    #             future.result()

    #     experiment.status = "completed"
    #     experiment.end_time = datetime.now()
    #     experiment.save()  # Assuming you have an `experiment.save()` method



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


#example
# Create a pipeline instance
pipeline = Pipeline(name="Sample Pipeline", description="This is a sample pipeline")

# Add nodes to the pipeline
pipeline.add_node(load_data)
pipeline.add_node(process_data, dependencies=[load_data])

# Run the pipeline
input_artifacts = {}
config = {}

pipeline.run(input_artifacts, config)

# Shutdown the executor after running the pipeline
pipeline.shutdown()



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
