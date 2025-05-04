import argparse
from pipelines.training_pipeline import TrainingPipeline

def main():
    # Set up parser
    parser = argparse.ArgumentParser(description="Running the pipelines")

    parser.add_argument("--local", action="store_true", help="Use local file storage")
    parser.add_argument("--cloud", action="store_true", help="Use AWS cloud storage")
    parser.add_argument(
        "--localstack", action="store_true", help="Use LocalStack for testing"
    )
    parser.add_argument(
        "--data", default="data/external/sample.csv", help="Path to dataset"
    )

    # Parse arguments
    args = parser.parse_args()
    # config_file = "config/local.yaml"  # default

    if args.local:
        config_file = "config/local.yaml"
    elif args.cloud:
        config_file = "config/cloud.yaml"
    elif args.localstack:
        config_file = "config/localstack.yaml"
    else:
        print("Please define config file")  # Default to local

    # Initialize pipeline
    pipeline = TrainingPipeline(
        data_path=args.data,
        config_file=config_file,
    )

    # Run pipeline
    results = pipeline.run()


if __name__ == "__main__":
    main()
