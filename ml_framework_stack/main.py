# 16. Example of running a pipeline with MLOps stack
def run_example():
    """Run the example pipeline with an MLOps stack."""
    # Create MLOps stack
    stack = Stack()
    
    # Add experiment tracking
    tracker = ExperimentTracker({
        'tracking_uri': 'sqlite:///mlflow.db',
        'experiment_name': 'example'
    })
    stack.add_component('tracking', tracker)
    
    # Add model registry
    registry = ModelRegistry({})
    stack.add_component('registry', registry)
    
    # Add storage
    storage = StorageComponent({'type': 'local'})
    stack.add_component('storage', storage)
    
    # Add deployment
    deployment = DeploymentComponent({'type': 'bentoml'})
    stack.add_component('deployment', deployment)
    
    # Create data catalog
    catalog = DataCatalog()
    catalog.add('data_path', CSVDataset('data/01_raw/data.csv'))
    catalog.add('raw_data', ParquetDataset('data/02_intermediate/raw_data.parquet'))
    catalog.add('processed_data', ParquetDataset('data/03_primary/processed_data.parquet'))
    catalog.add('model', PickleDataset('data/06_models/model.pkl'))
    
    # Create context with parameters
    context = Context(
        catalog=catalog,
        stack=stack,
        params={
            'test_size': 0.2,
            'n_estimators': 100,
            'random_state': 42
        }
    )
    
    # Get pipeline
    pipeline = example_pipeline()
    
    # Run pipeline
    results = context.run(pipeline)
    
    # Deploy model
    model = results.get('model')
    if model:
        stack.deployment.deploy_model(model, 'example_model', version='1.0.0')
        
    return results


# Example pipeline visualization function
def visualize_pipeline(pipeline):
    """Create a visualization of a pipeline."""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.DiGraph()
        
        # Add nodes
        for node in pipeline.nodes:
            G.add_node(node.name)
            
        # Add edges
        for node in pipeline.nodes:
            for input_name in node.inputs.values():
                for other_node in pipeline.nodes:
                    if input_name in other_node.outputs.values():
                        G.add_edge(other_node.name, node.name, label=input_name)
        
        # Draw the graph
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=10, font_weight='bold')
        
        # Draw edge labels
        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title("Pipeline Visualization")
        plt.savefig("pipeline_graph.png")
        plt.close()
        
        return "pipeline_graph.png"
    except ImportError:
        print("NetworkX and/or matplotlib not installed. Visualization skipped.")
        return None


# Main function to showcase the framework
if __name__ == "__main__":
    print("ML Framework Example")
    results = run_example()
    print("Pipeline execution results:", results)
