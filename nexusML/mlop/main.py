from core.pipeline import TrainingPipeline
from config import load_config

def main():
    config = load_config('config/config.yml')
    pipeline = TrainingPipeline()
    results = pipeline.run(config)
    print(f"Pipeline Results: {results}")

if __name__ == '__main__':
    main()