# Setup Guide

This guide shows how to deploy this exemplar to your Databricks workspace, and how to use it as a starting point for your own project.

## Prerequisites

Before you begin:

1. **Databricks CLI** installed ([docs](https://docs.databricks.com/aws/en/dev-tools/cli/install.html))
2. **Workspace authentication** configured (see below)
3. **Unity Catalog** enabled on your workspace
4. **Model Serving** enabled (check workspace settings)
5. **Serverless Compute** available in your region
6. **Permissions** to create jobs, models, serving endpoints, and tables

### Configure Workspace Authentication

The bundle needs to know which workspace to deploy to. Choose one method:

**Option A: CLI Profile (recommended for multiple workspaces)**

```bash
# Create a named profile with OAuth authentication
databricks auth login --host https://your-workspace.cloud.databricks.com --profile my-profile

# Verify it works
databricks current-user me --profile my-profile

# Use with bundle commands
databricks bundle deploy --profile my-profile
```

**Option B: Environment Variable (simpler for single workspace)**

```bash
# Set the workspace host
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com

# Authenticate
databricks auth login

# Verify
databricks current-user me

# Bundle commands will use DATABRICKS_HOST automatically
databricks bundle deploy
```

**Option C: Default Profile**

If you have a `[DEFAULT]` profile in `~/.databrickscfg`, bundle commands will use it automatically when no `--profile` is specified.

---

## Option A: Deploy the Exemplar Directly

Use this option to explore the exemplar without modification.

### Step 1: Choose Your Catalog and Schema

The exemplar writes tables and models to Unity Catalog. You need a catalog and schema with write permissions.

Variables are **required** and must be provided via `--var` flag or environment variables.

### Step 2: Validate

```bash
cd exemplars/ml-model-serving

databricks bundle validate --profile YOUR_PROFILE \
  --var="catalog=your_catalog,schema=your_schema"
```

You should see:

```
Name: ml-model-serving
Target: dev
Workspace:
  Host: https://your-workspace.cloud.databricks.com
  ...

Validation OK!
```

### Step 3: Deploy

```bash
databricks bundle deploy --profile YOUR_PROFILE \
  --var="catalog=your_catalog,schema=your_schema"
```

This creates three jobs: a training pipeline, a batch inference job, and a monitoring job.

### Step 4: Run the Training Pipeline

```bash
databricks bundle run ml_training_pipeline --profile YOUR_PROFILE \
  --var="catalog=your_catalog,schema=your_schema"
```

This runs three tasks sequentially:
1. **train** — Loads taxi data, engineers features, trains a model, logs to MLflow (~3-5 min)
2. **validate** — Registers model in Unity Catalog, compares against Champion (~1-2 min)
3. **deploy** — Creates the serving endpoint and promotes to Champion (~5-10 min first run)

Monitor progress in the Databricks UI under **Workflows > Jobs**.

### Step 5: Run Batch Inference

```bash
databricks bundle run ml_batch_inference --profile YOUR_PROFILE \
  --var="catalog=your_catalog,schema=your_schema"
```

Scores the feature table through the serving endpoint using `ai_query()`.

### Step 6: Run Monitoring

```bash
databricks bundle run ml_monitoring --profile YOUR_PROFILE \
  --var="catalog=your_catalog,schema=your_schema"
```

Unpacks inference table payloads, joins with ground truth, and creates a Lakehouse Monitor. View the auto-generated dashboard in **Catalog Explorer > `{model_name}_inference_processed` table > Quality tab**.

**Tip:** To avoid repeating `--var` on every command, set environment variables:

```bash
export BUNDLE_VAR_catalog=your_catalog
export BUNDLE_VAR_schema=your_schema

# Now you can run without --var
databricks bundle deploy --profile YOUR_PROFILE
databricks bundle run ml_training_pipeline --profile YOUR_PROFILE
```

### Step 7: Verify

```bash
# Check the serving endpoint
databricks serving-endpoints get your_schema_fare_predictor_model_endpoint

# Check tables
databricks tables list --catalog-name your_catalog --schema-name your_schema
```

---

## Option B: Use as a Template for Your Project

Use this option to create a new project based on this exemplar.

### Step 1: Copy the Exemplar

```bash
# Using the bootstrap script (recommended)
./shared/scripts/init-from-exemplar.sh ml-model-serving my-ml-project

# Or manually
cp -r exemplars/ml-model-serving ~/projects/my-ml-project
cd ~/projects/my-ml-project
git init
```

### Step 2: Rename the Bundle

Edit `databricks.yml`:

```yaml
bundle:
  name: my-ml-project  # <- Your project name

variables:
  catalog:
    default: your_catalog
  schema:
    default: your_schema
  model_name:
    default: my_model
```

### Step 3: Update the Training Code

Edit `src/train.py`:
- Change the data source from `samples.nyctaxi.trips`
- Update feature engineering for your domain
- Adjust model algorithm and hyperparameters

Edit `src/features.py`:
- Update `FEATURE_COLUMNS`, thresholds, and business rules
- Tests import from this file, so they'll stay in sync automatically

### Step 4: Update Inference and Monitoring

Edit `src/batch_inference.py`:
- Update the `named_struct()` in `ai_query()` to match your feature columns

Edit `src/monitoring.py`:
- Update `request_fields` and `response_field` to match your model's input/output schema

### Step 5: Deploy Your Project

```bash
cd ~/projects/my-ml-project
databricks bundle deploy --profile YOUR_PROFILE
```

---

## Configuration Reference

### Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `catalog` | **Yes** | Unity Catalog name | — |
| `schema` | **Yes** | Schema for tables and models | — |
| `model_name` | No | Registered model name | `fare_predictor_model` |

Set variables using one of these methods:

```bash
# Method 1: Command line flag
--var="catalog=my_catalog,schema=my_schema"

# Method 2: Environment variables
export BUNDLE_VAR_catalog=my_catalog
export BUNDLE_VAR_schema=my_schema
```

### Targets

| Target | Mode | Description |
|--------|------|-------------|
| `dev` | development | Development mode (default) |
| `prod` | production | Production mode |

Deploy to production:

```bash
databricks bundle deploy --target prod --profile YOUR_PROFILE \
  --var="catalog=prod_catalog,schema=prod_schema"
```

### What Gets Created

| Resource | Name Pattern |
|----------|-------------|
| Feature table | `{catalog}.{schema}.feature_table` |
| Predictions table | `{catalog}.{schema}.predictions` |
| Inference table | `{catalog}.{schema}.{endpoint}_payload` (auto) |
| Processed table | `{catalog}.{schema}.{model_name}_inference_processed` |
| Registered model | `{catalog}.{schema}.{model_name}` |
| MLflow experiment | `/Shared/{catalog}_{schema}_{model_name}` |
| Serving endpoint | `{schema}_{model_name}_endpoint` |
| Lakehouse Monitor | Attached to processed table |

---

## Common Customizations

### Enable Scheduled Jobs

The batch inference and monitoring jobs are deployed paused. To enable scheduling, edit the YAML:

In `resources/batch_job.yml`:
```yaml
schedule:
  quartz_cron_expression: "0 0 */6 * * ?"  # Every 6 hours
  timezone_id: UTC
  pause_status: UNPAUSED  # <- Change from PAUSED
```

In `resources/monitoring_job.yml`:
```yaml
schedule:
  quartz_cron_expression: "0 0 8 * * ?"  # Daily at 8 AM UTC
  timezone_id: UTC
  pause_status: UNPAUSED  # <- Change from PAUSED
```

### Change Endpoint Size

In `src/deploy.py`, modify the `ServedEntityInput`:
```python
served_entity = ServedEntityInput(
    entity_name=uc_model_name,
    entity_version=str(model_version),
    workload_size="Large",       # Small, Medium, or Large
    scale_to_zero_enabled=True,
)
```

### Add Email Notifications

In any job YAML:
```yaml
email_notifications:
  on_failure:
    - team@example.com
```

---

## Troubleshooting

### "Catalog not found" or "Schema not found"

Create the schema if it doesn't exist:

```sql
CREATE CATALOG IF NOT EXISTS your_catalog;
CREATE SCHEMA IF NOT EXISTS your_catalog.your_schema;
```

Or grant yourself access:

```sql
GRANT ALL PRIVILEGES ON CATALOG your_catalog TO `your-email@example.com`;
```

### "Model Serving not enabled"

Contact your workspace admin to enable Model Serving in workspace settings.

### "Serverless compute not available"

Serverless may not be available in all regions. See [supported regions](https://docs.databricks.com/en/compute/serverless.html).

### "Permission denied creating endpoint"

Request `CAN_CREATE_SERVING_ENDPOINTS` permission from your workspace admin.

### "Cold start timeout during validation"

The validate task includes retry logic with exponential backoff. If timeouts persist, check model artifact size and dependencies in the endpoint logs.

### Deploy task skipped — "Model not better than Champion"

This is expected behavior on re-runs with the same data. The Champion/Challenger gate prevents unnecessary redeployments. To force a new deployment, delete the Champion alias:

```bash
databricks api delete /api/2.0/mlflow/registered-models/alias \
  --json '{"name": "your_catalog.your_schema.fare_predictor_model", "alias": "Champion"}'
```

### SDK import errors in monitoring job

The pre-installed `databricks-sdk` in the Databricks Runtime may be too old. The monitoring job pins `databricks-sdk>=0.68.0` in its environment spec. If you see import errors, verify the dependency is listed in `resources/monitoring_job.yml`.

---

## Cleanup

To remove all deployed resources:

```bash
databricks bundle destroy --profile YOUR_PROFILE \
  --var="catalog=your_catalog,schema=your_schema"
```

This removes the jobs but **does not delete** the serving endpoint, tables, models, or monitoring resources. To fully clean up:

```bash
# Delete serving endpoint
databricks serving-endpoints delete your_schema_fare_predictor_model_endpoint

# Delete registered model
databricks unity-catalog models delete \
  --full-name your_catalog.your_schema.fare_predictor_model
```

```sql
-- Drop tables and schema
DROP SCHEMA IF EXISTS your_catalog.your_schema CASCADE;
```
