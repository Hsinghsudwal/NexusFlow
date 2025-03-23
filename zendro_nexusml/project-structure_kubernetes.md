mlops-customer-churn/
├── .github/
│   └── workflows/
│       ├── ci.yaml
│       └── cd.yaml
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.training
│   ├── Dockerfile.monitoring
│   └── Dockerfile.data-pipeline
├── k8s/
│   ├── namespaces/
│   │   ├── mlops-dev.yaml
│   │   ├── mlops-staging.yaml
│   │   └── mlops-prod.yaml
│   ├── kubeflow/
│   │   ├── kustomization.yaml
│   │   ├── pipeline.yaml
│   │   └── notebook.yaml
│   ├── monitoring/
│   │   ├── prometheus.yaml
│   │   ├── grafana.yaml
│   │   └── alertmanager.yaml
│   └── services/
│       ├── model-api.yaml
│       ├── mlflow.yaml
│       └── postgres.yaml
├── kubeflow_pipelines/
│   ├── pipeline_definition.py
│   ├── components/
│   │   ├── data_prep/
│   │   │   ├── src/
│   │   │   │   └── data_prep.py
│   │   │   └── component.yaml
│   │   ├── training/
│   │   │   ├── src/
│   │   │   │   └── train.py
│   │   │   └── component.yaml
│   │   ├── evaluation/
│   │   │   ├── src/
│   │   │   │   └── evaluate.py
│   │   │   └── component.yaml
│   │   └── deployment/
│   │       ├── src/
│   │       │   └── deploy.py
│   │       └── component.yaml
│   └── workflows/
│       ├── training_pipeline.py
│       └── retraining_pipeline.py
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── routers/
│   │   │   ├── predict.py
│   │   │   └── health.py
│   │   └── services/
│   │       ├── prediction.py
│   │       └── model_loader.py
│   ├── data/
│   │   ├── ingest.py
│   │   ├── preprocess.py
│   │   ├── validate.py
│   │   └── feature_store.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── model.py
│   └── monitoring/
│       ├── drift_detector.py
│       ├── metrics_collector.py
│       └── alert_manager.py
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── feature_engineering.ipynb
│   └── model_evaluation.ipynb
├── tests/
│   ├── unit/
│   │   ├── test_model.py
│   │   ├── test_preprocessing.py
│   │   └── test_api.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_database.py
│   └── e2e/
│       └── test_deployment.py
├── local/
│   ├── docker-compose.yaml
│   ├── postgres/
│   │   └── init.sql
│   └── localstack/
│       └── init.sh
├── config/
│   ├── dev.yaml
│   ├── staging.yaml
│   └── prod.yaml
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
└── README.md
