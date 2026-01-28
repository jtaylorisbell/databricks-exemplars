# ML Model Serving

An end-to-end machine learning workflow demonstrating MLflow experiment tracking, Unity Catalog model registry, and real-time model serving with serverless compute.

## Use Case

Predict NYC taxi fare amounts using trip distance and temporal features. Uses `samples.nyctaxi.trips` as source data (21,932 records).

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Train     │───▶│   Validate   │───▶│    Deploy    │───▶│   Endpoint   │
│  (MLflow)    │    │  (Compare)   │    │  (Promote)   │    │  (Serving)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                                       │
       ▼                   ▼                                       ▼
┌──────────────┐    ┌──────────────┐                        ┌──────────────┐
│   Feature    │    │   Champion   │                        │    Batch     │
│    Table     │    │   vs New     │                        │  Inference   │
└──────────────┘    └──────────────┘                        └──────────────┘
                                                                   │
                                                                   ▼
                                                            ┌──────────────┐
                                                            │  Monitoring  │
                                                            │   (Drift)    │
                                                            └──────────────┘
```

## Design Principles

### Deploy Code, Not Models

This exemplar follows the "deploy code, not models" pattern. The serving endpoint is **not** defined as a DAB resource. Instead, the `deploy.py` notebook programmatically creates or updates the endpoint when a new model outperforms the current Champion.

Benefits:
- Full control over deployment logic and validation gates
- Model version explicitly set (not alias-based auto-update)
- Endpoint only updates when validation passes

### Serverless Compute

All jobs use serverless compute via environment specifications. No cluster configurations needed.

### Champion/Challenger Pattern

- New models are registered with a `Challenger` alias
- Validation compares Challenger vs current `Champion` (if exists)
- Only models that outperform Champion get promoted and deployed

## What's Included

| Job | Description | Tasks |
|-----|-------------|-------|
| `ml_training_pipeline` | Train and deploy workflow | train → validate → deploy |
| `ml_batch_inference` | Scheduled batch scoring | batch_predict (every 6 hours) |
| `ml_monitoring` | Drift detection | monitor_drift (daily) |

### Tables Created

| Table | Purpose |
|-------|---------|
| `feature_table` | Features for training and inference |
| `training_baseline` | Statistical baseline for drift detection |
| `predictions` | Batch inference results with actuals |
| `model_monitoring` | Drift metrics and alerts |

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Model Serving enabled
- Permissions to create jobs, models, and endpoints
- Access to `samples.nyctaxi.trips` dataset

## Quick Start

### 1. Configure CLI Authentication

```bash
# Authenticate with your workspace
databricks auth login --host https://your-workspace.cloud.databricks.com
```

Or use a named profile:
```bash
databricks auth login --host https://your-workspace.cloud.databricks.com --profile my-profile
```

### 2. Deploy the Bundle

```bash
cd exemplars/ml-model-serving

# Deploy with required variables
databricks bundle deploy \
  --var="catalog=my_catalog,schema=ml_serving,model_name=fare_predictor"
```

Or with a specific profile:
```bash
databricks bundle deploy \
  --profile my-profile \
  --var="catalog=my_catalog,schema=ml_serving,model_name=fare_predictor"
```

### 3. Run the Training Pipeline

```bash
databricks bundle run ml_training_pipeline \
  --var="catalog=my_catalog,schema=ml_serving,model_name=fare_predictor"
```

### 4. Test the Endpoint

After training completes, the endpoint will be available:

```bash
# Get endpoint URL
databricks serving-endpoints get ml_serving_fare_predictor_endpoint

# Test prediction
curl -X POST "https://<workspace>/serving-endpoints/<endpoint>/invocations" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [{
      "trip_distance": 3.5,
      "pickup_hour": 17,
      "pickup_dayofweek": 3,
      "pickup_month": 6,
      "is_weekend": 0,
      "is_rush_hour": 1
    }]
  }'
```

## Configuration

### Required Variables

All variables must be provided via `--var` flag:

| Variable | Description | Example |
|----------|-------------|---------|
| `catalog` | Unity Catalog name | `my_catalog` |
| `schema` | Schema for tables and models | `ml_serving` |
| `model_name` | Registered model name | `fare_predictor` |

### Passing Variables

```bash
# Single command
--var="catalog=my_catalog,schema=ml_serving,model_name=fare_predictor"

# Or multiple flags
--var="catalog=my_catalog" --var="schema=ml_serving" --var="model_name=fare_predictor"
```

## Pipeline Details

### Training Task

1. Loads `samples.nyctaxi.trips` data
2. Engineers features (trip distance, time-based features)
3. Trains GradientBoostingRegressor with hyperparameter search
4. Logs metrics and model to MLflow
5. Saves feature table and training baseline

### Validation Task

1. Registers model to Unity Catalog
2. Compares MAE against current Champion (if exists)
3. Tests model loading with cold-start retry logic
4. Sets `Challenger` alias on new version
5. Passes validation result to deploy task

### Deploy Task

1. Checks if model outperforms Champion
2. Promotes to `Champion` alias if better
3. Creates or updates serving endpoint
4. Verifies deployment with test prediction

### Batch Inference Task

1. Loads Champion model via alias
2. Scores feature table
3. Appends predictions with timestamps
4. Calculates batch inference metrics

### Monitoring Task

1. Loads training baseline statistics
2. Analyzes recent predictions (last 24h)
3. Calculates drift z-scores
4. Writes alerts to monitoring table
5. Flags `should_retrain` when thresholds exceeded

## Drift Detection

The monitoring job tracks:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Prediction drift | z-score > 2.0 | Mean prediction shift |
| Feature drift | z-score > 2.0 | Input distribution shift |
| MAE degradation | > $5.00 | Accuracy decline |

When alerts trigger, `should_retrain=true` is recorded in the monitoring table.

## Testing

```bash
# Install dev dependencies
uv sync --extra dev

# Run unit tests
uv run pytest tests/
```

## Files

```
ml-model-serving/
├── databricks.yml          # Bundle configuration
├── resources/
│   ├── training_job.yml    # Training pipeline job
│   ├── batch_job.yml       # Batch inference job
│   └── monitoring_job.yml  # Monitoring job
├── src/
│   ├── train.py            # Model training notebook
│   ├── validate.py         # Model validation notebook
│   ├── deploy.py           # Endpoint deployment notebook
│   ├── batch_inference.py  # Batch scoring notebook
│   └── monitoring.py       # Drift monitoring notebook
├── tests/
│   └── test_features.py    # Unit tests
├── pyproject.toml
└── README.md
```

## Learn More

- [MLflow on Databricks](https://docs.databricks.com/en/mlflow/index.html)
- [Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
- [Unity Catalog Model Registry](https://docs.databricks.com/en/mlflow/model-registry.html)
- [Lakehouse Monitoring](https://docs.databricks.com/en/lakehouse-monitoring/index.html)
