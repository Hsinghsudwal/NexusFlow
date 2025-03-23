from core.pipeline import TrainingPipeline
from config import load_config

def main():
    config = load_config('config/config.yml')
    pipeline = TrainingPipeline(config)
    results = pipeline.run()
    print("Pipeline execution results:", results)

if __name__ == "__main__":
    main()