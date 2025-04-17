# Project Template: NexStream

This nexstream MLOps framework is a production-ready MLOps with best practices for machine learning model development, deployment, and monitoring.

### Core Architecture

- Function- passes input and return outputs
- Pipeline Construction: Pipelines are created by connecting Function together
- Config: Handles inputs
- Artifacts_store: Handle output storage
- Execution Context: Provides the environment for pipeline execution

## MLOps Stack: Integrates various MLOps components

This framework includes a modular stack system with components like:

- Experiment Tracking: Integration with MLflow for metrics and parameters
- Model Registry: Version control for ML models
- Storage: Flexible storage backends (local, S3, GCS)
- Deployment: Model serving tools: (Flask, Fastapi, Streamlit)
- Monitoring: Drift-garfana, Prometheus
- Retraining: trigger based on drift

### Key Features

* Declarative Pipeline Definition: Clear separation of data and processing logic
* Dataset Abstraction: Support for various data formats (CSV, Parquet, Pickle)
* CLI Interface: Command line tools for running pipelines
* Environment Configuration: Configuration management for different environments


## Getting Started

### Prerequisites

- Python 3.8+
- Docker
- Make

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### Development

1. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

2. Run tests:
```bash
make test
```

3. Run linting:
```bash
make lint
```

### Model Training

```bash
python -m ml_pipeline.train
```

### Model Serving

```bash
python -m serving.app
```

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment. See `.github/workflows/` for pipeline configurations.

## Documentation

Detailed documentation is available in the `docs/` directory.

## License

[Your License]

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
