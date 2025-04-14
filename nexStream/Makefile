.PHONY: help install test lint format clean train serve docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install    - Install project dependencies"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linting"
	@echo "  format     - Format code"
	@echo "  clean      - Clean up build artifacts"
	@echo "  train      - Train the model"
	@echo "  serve      - Serve the model"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

lint:
	flake8 .
	black . --check
	isort . --check-only

format:
	black .
	isort .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

train:
	python -m ml_pipeline.train

serve:
	python -m serving.app

docker-build:
	docker build -t mlops-app .

docker-run:
	docker run -p 8000:8000 mlops-app

# Data processing
process-data:
	python -m ml_pipeline.data_processing

# Model evaluation
evaluate:
	python -m ml_pipeline.evaluate

# Documentation
docs:
	cd docs && make html

# Development setup
setup-dev:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements-dev.txt
	pre-commit install
