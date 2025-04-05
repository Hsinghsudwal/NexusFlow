import argparse
from training_pipeline import TrainingPipeline

def main():

    parser = argparse.ArgumentParser(description="Running the pipelines")
    
    # Mutually exclusive group for local/cloud execution
    execution_group = parser.add_mutually_exclusive_group(required=True)
    execution_group.add_argument("--local", action="store_true", help="Execute pipeline locally")
    execution_group.add_argument("--cloud", action="store_true", help="Execute pipeline in cloud")
    
    # Parse arguments
    args = parser.parse_args()


    if args.local:
        execution_mode = "local"
    else:
        execution_mode = "cloud"
    
    # Initialize pipeline
    pipeline = TrainingPipeline( 
        execution_mode=execution_mode
    )
    
    # Run pipeline
    pipeline.run()


if __name__ == "__main__":
    main()


```python run.py --local```