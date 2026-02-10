# End-to-End Machine Learning Workflow

An end-to-end machine learning workflow demonstrating MLflow experiment tracking, Unity Catalog model registry, real-time model serving, and automated monitoring with Data Quality Monitoring.

## Use Case

Predict NYC taxi fare amounts using trip distance and temporal features. Uses `samples.nyctaxi.trips` as source data.

## Architecture

```
Training Pipeline (on-demand)
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Train     │───▶│   Validate   │───▶│    Deploy     │
│   (MLflow)   │    │  (Champion   │    │  (Endpoint +  │
│              │    │  vs Chall.)  │    │   Promote)    │
└──────┬───────┘    └──────────────┘    └──────┬───────┘
       │                                       │
       ▼                                       ▼
 Feature Table                         Serving Endpoint
                                       (route-optimized,
                                        scale-to-zero)
                                               │
               ┌───────────────────────────────┤
               │                               │
               ▼                               ▼
        Batch Inference                 AI Gateway
        (every 6 hours)              Inference Table
        uses ai_query()            (auto-logged payloads)
               │                               │
               ▼                               ▼
        Predictions Table            Monitoring (daily)
        (convenience layer)          Data Quality Monitoring
                                     (drift + quality)
```

## Key Design Decisions

### Deploy Code, Not Models

The serving endpoint is **not** defined as a DAB resource. Instead, `deploy.py` programmatically creates or updates the endpoint when a new model outperforms the current Champion. This gives full control over deployment logic, validation gates, and rollback.

### ai_query() for Batch Inference

Batch inference uses `ai_query()` to route predictions through the serving endpoint rather than loading the model onto the driver. This means every prediction — batch or real-time — flows through the same endpoint and is automatically logged to the AI Gateway inference table.

### Data Quality Monitoring

Instead of custom drift detection code, this exemplar uses Databricks Data Quality Monitoring which automatically tracks:

- **Inference volume** over time
- **Model quality** metrics (MAE, RMSE, MSE, MAPE, R2)
- **Prediction drift** detection
- **Feature drift** across all input columns

The monitoring job unpacks inference table payloads into a structured Delta table, joins with ground truth, and creates a Data Quality Monitor that generates dashboards automatically.

After running the monitoring job, the auto-generated dashboard (accessible via **Catalog Explorer > your table > Quality tab**) shows:
- **Inference volume** over time (e.g., 48K inferences in the last window)
- **Performance metrics**: MAE, RMSE, MSE, MAPE, and R2 score
- **Performance over time** charts tracking all metrics across time windows
- **Prediction drift** detection with statistical analysis

### Champion/Challenger Pattern

- New models are registered with a `Challenger` alias
- Validation compares Challenger MAE against the current `Champion`
- Only models that outperform Champion get deployed
- Champion alias is set **after** the endpoint is verified healthy, preventing inconsistent state if deployment fails

### Serverless Compute

All jobs use serverless compute. No cluster configurations needed.

## What's Included

### Jobs

| Job | Schedule | Description |
|-----|----------|-------------|
| `ml_training_pipeline` | On-demand | Train → Validate → Deploy (multi-task) |
| `ml_batch_inference` | Every 6 hours (paused) | Score feature table via `ai_query()` |
| `ml_monitoring` | Daily at 8 AM UTC (paused) | Unpack inference table, refresh Data Quality Monitor |

### Tables

| Table | Created By | Purpose |
|-------|-----------|---------|
| `feature_table` | train.py | Features for training and batch inference |
| `predictions` | batch_inference.py | Structured predictions (convenience layer — raw data also in inference table) |
| `{model_name}_inference_processed` | monitoring.py | Unpacked inference records with ground truth, monitored by Data Quality Monitoring |
| `{endpoint}_payload` | AI Gateway | Auto-logged raw request/response payloads |

### Serving Endpoint

| Setting | Value |
|---------|-------|
| Workload size | Medium |
| Scale to zero | Enabled |
| Route optimization | Enabled |
| Inference table logging | Enabled (via AI Gateway) |

## Quick Start

```bash
cd exemplars/ml-model-serving

# 1. Deploy
databricks bundle deploy \
  --var="catalog=my_catalog" \
  --var="schema=my_schema"

# 2. Train, validate, and deploy the model
databricks bundle run ml_training_pipeline \
  --var="catalog=my_catalog" \
  --var="schema=my_schema"

# 3. Run batch inference
databricks bundle run ml_batch_inference \
  --var="catalog=my_catalog" \
  --var="schema=my_schema"

# 4. Run monitoring
databricks bundle run ml_monitoring \
  --var="catalog=my_catalog" \
  --var="schema=my_schema"
```

See [SETUP.md](SETUP.md) for detailed setup instructions, authentication options, and troubleshooting.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `catalog` | Unity Catalog name | (required) |
| `schema` | Schema for tables and models | (required) |
| `model_name` | Registered model name | `fare_predictor_model` |

## Pipeline Details

### Train (`src/train.py`)

1. Loads `samples.nyctaxi.trips` and engineers features (trip distance, pickup hour, day of week, month, weekend flag, rush hour flag)
2. Saves the feature table to Unity Catalog
3. Trains a `GradientBoostingRegressor` with hyperparameter grid search (GridSearchCV)
4. Logs parameters, metrics, and model artifact to MLflow
5. Passes the MLflow run ID to the validate task

### Validate (`src/validate.py`)

1. Registers the trained model to Unity Catalog from the MLflow run
2. Compares test MAE against the current Champion model (if one exists)
3. Tests model loading with exponential backoff retry logic for cold starts
4. Sets `Challenger` alias on the new version
5. Passes `is_better`, `model_version`, and `test_mae` to the deploy task

### Deploy (`src/deploy.py`)

1. Skips deployment if the model doesn't outperform Champion
2. Creates or updates the serving endpoint with the new model version
3. Enables AI Gateway inference table logging for automatic request/response capture
4. Verifies the endpoint is healthy with a test prediction
5. Promotes to `Champion` alias only after verification succeeds

### Batch Inference (`src/batch_inference.py`)

1. Queries the serving endpoint for the current model version
2. Scores the entire feature table using `ai_query()` with `named_struct()`
3. Writes predictions to a convenience table (raw data is also auto-logged to the inference table)

### Monitoring (`src/monitoring.py`)

1. Reads raw payloads from the AI Gateway inference table
2. Unpacks JSON request/response using a consolidation UDF that handles all Model Serving formats
3. Joins with the feature table to attach ground truth (actual fare amounts)
4. Writes to a processed Delta table with Change Data Feed enabled
5. Creates or refreshes a Data Quality Monitor configured for regression

## Testing

```bash
# Install dev dependencies
uv sync --extra dev

# Run unit tests
uv run pytest tests/
```

Tests import business rules from `src/features.py` (the single source of truth for feature constants and transformation logic) to validate that feature engineering rules are correct.

## Files

```
ml-model-serving/
├── databricks.yml              # Bundle configuration
├── README.md
├── SETUP.md                    # Detailed setup and troubleshooting
├── pyproject.toml
├── resources/
│   ├── training_job.yml        # Train → Validate → Deploy pipeline
│   ├── batch_job.yml           # Scheduled batch inference
│   └── monitoring_job.yml      # Scheduled monitoring
├── src/
│   ├── train.py                # Model training notebook
│   ├── validate.py             # Model validation notebook
│   ├── deploy.py               # Endpoint deployment notebook
│   ├── batch_inference.py      # Batch scoring via ai_query()
│   ├── monitoring.py           # Inference table unpacking + Data Quality Monitor
│   └── features.py             # Shared feature constants and helpers
└── tests/
    └── test_features.py        # Unit tests for feature logic
```

## Learn More

- [Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
- [AI Gateway Inference Tables](https://docs.databricks.com/en/machine-learning/model-serving/inference-tables.html)
- [Data Quality Monitoring](https://docs.databricks.com/en/lakehouse-monitoring/index.html)
- [ai_query() Function](https://docs.databricks.com/en/sql/language-manual/functions/ai_query.html)
- [MLflow on Databricks](https://docs.databricks.com/en/mlflow/index.html)
- [Unity Catalog Model Registry](https://docs.databricks.com/en/mlflow/model-registry.html)
- [Route Optimization](https://docs.databricks.com/en/machine-learning/model-serving/route-optimization.html)
