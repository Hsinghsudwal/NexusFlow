nexusML : enchanced excutor worker
flyte : orchestration
ml_framework-stack : option=cloud,local,mlflow,registe model
zendro : focus with kedro catalog

This custom MLOps framework combines the structured approach of Kedro with the stack-based architecture of ZenML. Here's an overview of what I've created:
Core Architecture

Node System: Similar to Kedro, each processing step is defined as a Node with inputs and outputs
Pipeline Construction: Pipelines are created by connecting nodes together
Data Catalog: Manages datasets and handles I/O operations
Execution Context: Provides the environment for pipeline execution
MLOps Stack: Integrates various MLOps components

MLOps Stack Components
The framework includes a modular stack system with components for:

Experiment Tracking: Integration with MLflow for metrics and parameters
Model Registry: Version control for ML models
Storage: Flexible storage backends (local, S3, GCS)
Deployment: Model serving via BentoML or TensorFlow Serving

Key Features

Declarative Pipeline Definition: Clear separation of data and processing logic
Dataset Abstraction: Support for various data formats (CSV, Parquet, Pickle)
Visualization: Pipeline visualization capabilities
CLI Interface: Command line tools for running pipelines
Environment Configuration: Configuration management for different environments

Project Structure
The project follows a clean, modular structure with:

Core framework components in ml_framework/
Example project structure with designated folders for data, pipelines, and notebooks
Configuration management similar to Kedro's approach

To use this framework, you would define your pipeline steps as functions, wrap them in nodes, connect them into a pipeline, and execute them with a context that includes your MLOps stack components.




ZenML-Style Architecture

Step-Centric Design: Uses a step decorator pattern (like ZenML) instead of nodes, making pipeline definition more intuitive
Pipeline as a Sequence of Connected Steps: Steps connect to each other explicitly using the .connect() method
Stack-Based MLOps Components: Follows ZenML's stack concept for integrating MLOps tools
Materializers: Added support for typed artifacts with specialized serializers (materializers)

Key Framework Components

Steps: Individual processing units that can be connected together (replaces Kedro's nodes)
Pipeline: Collection of connected steps that form a workflow
Stack: Configuration of MLOps tools and integrations
Stack Registry: Central registry for managing stacks (similar to ZenML)
Materializers: Type-aware serialization mechanisms

MLOps Integration
The framework provides stack components for:

Experiment Tracking: MLflow integration
Artifact Store: Local, S3, or GCS storage
Orchestrator: Local execution, Airflow, or Prefect
Model Deployer: BentoML or MLflow deployment
