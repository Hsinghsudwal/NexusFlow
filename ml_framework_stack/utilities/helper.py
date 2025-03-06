# 12. Helper utilities
def pipeline_from_nodes(*nodes, name=None, tags=None):
    """Create a pipeline from a list of nodes."""
    return Pipeline(list(nodes), name=name, tags=tags or set())


def node_from_func(func, inputs=None, outputs=None, name=None, tags=None):
    """Create a node from a function."""
    return Node(func, inputs, outputs, name, tags)




# 8. Materializers - Similar to ZenML's concept
class Materializer(ABC):
    """Base class for materializers that handle artifact serialization."""
    
    @abstractmethod
    def save(self, artifact, path: str) -> None:
        """Save an artifact to a path."""
        pass
        
    @abstractmethod
    def load(self, path: str) -> Any:
        """Load an artifact from a path."""
        pass


class PandasMaterializer(Materializer):
    """Materializer for pandas DataFrames."""
    
    def save(self, df, path: str) -> None:
        """Save a DataFrame to a path."""
        import pandas as pd
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Determine format from file extension
        if path.endswith('.csv'):
            df.to_csv(path, index=False)
        elif path.endswith('.parquet'):
            df.to_parquet(path, index=False)
        elif path.endswith('.pkl'):
            df.to_pickle(path)
        else:
            # Default to parquet
            df.to_parquet(f"{path}.parquet", index=False)
            
    def load(self, path: str) -> Any:
        """Load a DataFrame from a path."""
        import pandas as pd
        
        # Determine format from file extension
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.parquet'):
            return pd.read_parquet(path)
        elif path.endswith('.pkl'):
            return pd.read_pickle(path)
        else:
            # Try to infer format or default to parquet
            try:
                return pd.read_parquet(f"{path}.parquet")
            except:
                raise ValueError(f"Cannot determine format for {path}")

