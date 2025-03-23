.PHONY: setup run train deploy monitor clean-data clean-models clean-all start-localstack stop-localstack

# Setup environment
setup:
	pip install -r requirements.txt
	mkdir -p data/raw data/processed models artifacts reports

# Start LocalStack
start-localstack:
	docker-compose up -d localstack

# Stop LocalStack
stop-localstack:
	docker-compose down

# Run full pipeline
run:
	python run.py --action full

# Train models only
train:
	python run.py --action train

# Deploy a model (requires model name)
deploy:
	python run.py --action deploy --model $(MODEL)

# Monitor deployed models
monitor:
	python run.py --action monitor --duration $(DURATION)

# Clean data files
clean-data:
	rm -rf data/processed/*
	rm -rf data/raw/*

# Clean model files
clean-models:
	rm -rf models/*
	rm -rf artifacts/*

# Clean all generated files
clean-all: clean-data clean-models
	rm -rf reports/*
	rm -f *.log
