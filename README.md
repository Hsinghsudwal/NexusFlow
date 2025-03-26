
## nexusFlow : orchestration, option=cloud,local,mlflow,registe model, enchanced excutor: sequential-parallel

This nexusFlow MLOps framework.

### Core Architecture

Function- passes input and return outputs
Pipeline Construction: Pipelines are created by connecting Function together
Config: Handles inputs
Artifacts_store: Handle output storage
Execution Context: Provides the environment for pipeline execution
MLOps Stack: Integrates various MLOps components

### MLOps Stack Components

The framework includes a modular stack system with components for:

- Experiment Tracking: Integration with MLflow for metrics and parameters
- Model Registry: Version control for ML models
- Storage: Flexible storage backends (local, S3, GCS)
- Deployment: Model serving tools:
- Monitoring: Drift-garfana, Prometheus
- Retraining: trigger based on drift

### Key Features

* Declarative Pipeline Definition: Clear separation of data and processing logic
* Dataset Abstraction: Support for various data formats (CSV, Parquet, Pickle)
* CLI Interface: Command line tools for running pipelines
* Environment Configuration: Configuration management for different environments




