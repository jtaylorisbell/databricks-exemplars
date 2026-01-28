# ML Model Serving

A production-ready end-to-end machine learning workflow with MLflow experiment tracking, model registry, and real-time model serving.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Training   │────▶│   MLflow    │────▶│   Model     │────▶│   Model     │
│    Data     │     │  Tracking   │     │  Registry   │     │  Serving    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                                       │
                    ┌──────┴──────┐                         ┌──────┴──────┐
                    │ Experiments │                         │  REST API   │
                    │   & Runs    │                         │  Endpoint   │
                    └─────────────┘                         └─────────────┘
```

## What's Included

- **Feature Engineering** — Feature table creation with Unity Catalog
- **Model Training** — Scikit-learn/XGBoost training with hyperparameter tuning
- **MLflow Tracking** — Experiment tracking with metrics and artifacts
- **Model Registry** — Version control and stage transitions
- **Model Serving** — Real-time inference endpoint
- **Inference Pipeline** — Batch scoring job

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- MLflow and Model Serving enabled
- Permissions to create endpoints and jobs

## Quick Start

```bash
# 1. Deploy to your workspace
databricks bundle deploy

# 2. Run the training job
databricks bundle run train_model

# 3. Deploy the model endpoint
databricks bundle run deploy_endpoint

# 4. Test the endpoint
curl -X POST https://<workspace>/serving-endpoints/<endpoint>/invocations \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -d '{"instances": [{"feature1": 1.0, "feature2": 2.0}]}'
```

## Configuration

Key variables in `databricks.yml`:

| Variable | Description | Default |
|----------|-------------|---------|
| `catalog` | Unity Catalog name | `main` |
| `schema` | Schema for tables | `ml_serving` |
| `model_name` | Registered model name | `my_model` |
| `endpoint_name` | Serving endpoint name | `my_model_endpoint` |

## Resources Created

- **Job**: `train_model` — Training pipeline with MLflow tracking
- **Job**: `batch_inference` — Scheduled batch scoring
- **Endpoint**: `my_model_endpoint` — Real-time serving endpoint
- **Tables**: `feature_table`, `predictions`

## Model Lifecycle

1. **Training** — Run experiments with MLflow tracking
2. **Registration** — Best model registered to Unity Catalog
3. **Validation** — Model validated before promotion
4. **Deployment** — Deployed to serving endpoint
5. **Monitoring** — Inference logs for drift detection

## Testing

```bash
# Run unit tests
uv run pytest tests/

# Test endpoint locally (mock)
uv run pytest tests/test_endpoint.py
```

## Learn More

- [MLflow on Databricks](https://docs.databricks.com/en/mlflow/index.html)
- [Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
- [Feature Store](https://docs.databricks.com/en/machine-learning/feature-store/index.html)
