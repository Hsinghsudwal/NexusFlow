
framework: Components

config_yaml <==> load_data ==>metadata, artifacts


load_data==> process ==> model ==> tune ==> evaluate ==> deploy ==> re-training + monitoring ==> streamlit ui

Pipelines: training <==> inference <==> trigger


Model Retraining System:

Added ml/models/retraining/ for retraining logic, triggers, and schedulers
Included retraining pipelines in ml/pipelines/retraining/
Added CI/CD workflow for automated retraining in .github/workflows/
Added retraining-specific configurations in configs/retraining/


Caching Framework:

Created dedicated cache/ directory for caching strategies
Added cache implementations for data, features, and inference results
Included infrastructure for distributed and local caching
Added cache policy management


Metadata Management:

Added ml/metadata/ for metadata tracking, schemas, and lineage
Included metadata store configurations and infrastructure
- Added Docker configuration for metadata services
- Added Kubernetes deployment configs for metadata services


Artifact Store:

Created dedicated artifacts/ directory for artifact management
Organized artifact versioning and promotion systems
Added infrastructure configurations for artifact storage
Included connectors for model, dataset, and feature artifacts