# Setup Guide

Detailed setup instructions for the ML Model Serving exemplar.

## Prerequisites

### Workspace Requirements

1. **Unity Catalog** — Must be enabled on your workspace
2. **Model Serving** — Must be enabled (check workspace settings)
3. **Serverless Compute** — Must be available in your region

### Permissions Required

- `CREATE CATALOG` or access to an existing catalog
- `CREATE SCHEMA` in your target catalog
- `CREATE MODEL` permission
- `CREATE SERVING ENDPOINT` permission
- Access to `samples.nyctaxi.trips` dataset

## Authentication Options

### Option A: Default Profile (Recommended)

```bash
# Login and set as default
databricks auth login --host https://your-workspace.cloud.databricks.com

# Verify authentication
databricks current-user me
```

### Option B: Named Profile

```bash
# Create a named profile
databricks auth login --host https://your-workspace.cloud.databricks.com --profile my-workspace

# Use with bundle commands
databricks bundle deploy --profile my-workspace --var="..."
```

### Option C: Environment Variable

```bash
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com

# Authentication will use OAuth browser flow
databricks bundle deploy --var="..."
```

## Step-by-Step Deployment

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd databricks-exemplars/exemplars/ml-model-serving
```

### 2. Validate the Bundle

```bash
databricks bundle validate \
  --var="catalog=your_catalog,schema=ml_serving,model_name=fare_predictor"
```

Expected output shows the resources that will be created.

### 3. Deploy

```bash
databricks bundle deploy \
  --var="catalog=your_catalog,schema=ml_serving,model_name=fare_predictor"
```

### 4. Run Training Pipeline

```bash
databricks bundle run ml_training_pipeline \
  --var="catalog=your_catalog,schema=ml_serving,model_name=fare_predictor"
```

Monitor progress in the Databricks UI under **Workflows > Jobs**.

### 5. Verify Resources

After training completes:

```bash
# Check model in Unity Catalog
databricks unity-catalog models list --catalog-name your_catalog --schema-name ml_serving

# Check serving endpoint
databricks serving-endpoints list

# Check tables
databricks tables list --catalog-name your_catalog --schema-name ml_serving
```

## Cleanup

To remove all deployed resources:

```bash
databricks bundle destroy \
  --var="catalog=your_catalog,schema=ml_serving,model_name=fare_predictor"
```

Note: This removes jobs but not:
- The serving endpoint (manually delete via UI or CLI)
- Tables and data (manually drop if needed)
- Registered models (manually delete via UI or CLI)

### Manual Cleanup

```bash
# Delete serving endpoint
databricks serving-endpoints delete ml_serving_fare_predictor_endpoint

# Delete model
databricks unity-catalog models delete your_catalog.ml_serving.fare_predictor

# Drop tables (in SQL)
# DROP TABLE your_catalog.ml_serving.feature_table;
# DROP TABLE your_catalog.ml_serving.training_baseline;
# DROP TABLE your_catalog.ml_serving.predictions;
# DROP TABLE your_catalog.ml_serving.model_monitoring;
```

## Using as a Template

### 1. Copy the Exemplar

```bash
cp -r exemplars/ml-model-serving ~/projects/my-ml-project
cd ~/projects/my-ml-project
git init
```

### 2. Customize for Your Use Case

**Update `databricks.yml`:**
```yaml
bundle:
  name: my-ml-project  # Change bundle name
```

**Modify `src/train.py`:**
- Change the data source from `samples.nyctaxi.trips`
- Update feature engineering for your domain
- Adjust model algorithm and hyperparameters

**Update feature columns** in all src files to match your schema.

### 3. Deploy Your Version

```bash
databricks bundle deploy \
  --var="catalog=my_catalog,schema=my_schema,model_name=my_model"
```

## Troubleshooting

### "Catalog not found"

Ensure you have access to the catalog:
```bash
databricks unity-catalog catalogs list
```

### "Model Serving not enabled"

Contact your workspace admin to enable Model Serving in workspace settings.

### "Serverless compute not available"

Serverless may not be available in all regions. Check [Databricks documentation](https://docs.databricks.com/en/compute/serverless.html) for supported regions.

### "Permission denied creating endpoint"

Request `CAN_CREATE_SERVING_ENDPOINTS` permission from your workspace admin.

### "Cold start timeout during validation"

The validate task includes retry logic for model loading. If timeouts persist:
1. Check model artifact size
2. Verify model dependencies are available
3. Review endpoint logs for errors
