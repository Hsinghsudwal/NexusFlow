# MLOps Customer Churn Project Structure

```
customer_churn_mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml                # GitHub Actions CI workflow
│       └── cd.yml                # GitHub Actions CD workflow
├── configs/
│   ├── params.yaml               # Project parameters
│   └── logging_config.yaml       # Logging configuration
├── data/
│   ├── raw/                      # Raw data storage
│   ├── processed/                # Processed data storage
│   └── test/                     # Test datasets
├── docker/
│   ├── Dockerfile                # Main application Dockerfile
│   ├── Dockerfile.api            # Model serving API Dockerfile
│   └── docker-compose.yml        # Docker compose for local development
├── docs/                         # Project documentation
├── infra/
│   ├── localstack/
│   │   └── setup.sh              # LocalStack setup script
│   └── terraform/                # IaC for cloud resources
├── logs/                         # Log files
├── mlflow/                       # MLflow artifacts
├── notebooks/                    # Jupyter notebooks for exploration
├── src/
│   ├── __init__.py
│   ├── main.py                   # Entry point for running the pipeline
│   ├── config.py                 # Configuration loader
│   ├── constants.py              # Project constants
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift_detection.py    # Data and model drift detection
│   │   ├── metrics.py            # Model performance metrics tracking
│   │   └── alerting.py           # Alerting system
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py     # Data ingestion pipeline
│   │   ├── data_validation.py    # Data validation pipeline
│   │   ├── data_preparation.py   # Data preparation/preprocessing
│   │   ├── feature_engineering.py # Feature engineering pipeline
│   │   ├── training.py           # Model training pipeline
│   │   ├── evaluation.py         # Model evaluation pipeline
│   │   ├── deployment.py         # Model deployment pipeline
│   │   ├── retraining.py         # Model retraining pipeline
│   │   └── registry.py           # Model registry interactions
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── data_schema.py        # Pydantic models for data validation
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── api.py                # FastAPI model serving API
│   │   └── middleware.py         # API middleware
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── local.py              # Local storage interface
│   │   └── s3.py                 # S3 storage interface (LocalStack)
│   └── utils/
│       ├── __init__.py
│       ├── artifact_manager.py   # Artifact management utilities
│       ├── logging_utils.py      # Logging utilities
│       ├── metadata.py           # Metadata tracking
│       └── db_utils.py           # Database utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Test fixtures
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── .gitignore
├── pyproject.toml                # Poetry dependencies
├── README.md
└── setup.py                      # Package installation
```
