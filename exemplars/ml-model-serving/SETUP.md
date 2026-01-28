# Setup: Using This Exemplar as a Template

Follow these steps to use this exemplar as a starting point for your ML project.

## 1. Copy the Exemplar

```bash
# Copy to your new project location
cp -r exemplars/ml-model-serving ~/projects/my-ml-project
cd ~/projects/my-ml-project

# Initialize git (optional but recommended)
git init
```

Or use the bootstrap script:

```bash
./shared/scripts/init-from-exemplar.sh ml-model-serving my-ml-project
```

## 2. Required Customizations

### Update `databricks.yml`

Open `databricks.yml` and update:

```yaml
bundle:
  name: my-ml-project  # Change from ml-model-serving

variables:
  catalog:
    default: your_catalog
  schema:
    default: your_schema
  model_name:
    default: your_model_name
  endpoint_name:
    default: your_model_endpoint
```

### Prepare Training Data

Ensure your training data exists in Unity Catalog:

```sql
-- Your feature table should exist
SELECT * FROM your_catalog.your_schema.training_data LIMIT 10;
```

## 3. Customize the Model

### Feature Engineering

Edit `src/feature_engineering.py` to define your features:

```python
# Define your feature transformations
features = (
    raw_data
    .withColumn("feature1", ...)
    .withColumn("feature2", ...)
)
```

### Model Training

Edit `src/train.py` to customize:

- Model algorithm (sklearn, XGBoost, PyTorch, etc.)
- Hyperparameter search space
- Evaluation metrics
- Training data preprocessing

### Inference Logic

Edit `src/inference.py` for:

- Input preprocessing
- Output postprocessing
- Custom scoring logic

## 4. Deploy

```bash
# Validate your configuration
databricks bundle validate

# Deploy resources
databricks bundle deploy

# Run training
databricks bundle run train_model
```

## 5. Create Serving Endpoint

After training completes and the model is registered:

```bash
# Deploy the serving endpoint
databricks bundle run deploy_endpoint

# Or create manually via CLI
databricks serving-endpoints create --json '{
  "name": "your_model_endpoint",
  "config": {
    "served_models": [{
      "model_name": "your_catalog.your_schema.your_model_name",
      "model_version": "1",
      "workload_size": "Small",
      "scale_to_zero_enabled": true
    }]
  }
}'
```

## 6. Test the Endpoint

```bash
# Get endpoint status
databricks serving-endpoints get your_model_endpoint

# Test inference
curl -X POST "https://<workspace>/serving-endpoints/your_model_endpoint/invocations" \
  -H "Authorization: Bearer $(databricks auth token)" \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"feature1": 1.0, "feature2": 2.0}]}'
```

## 7. Set Up Batch Inference (Optional)

The `batch_inference` job scores data on a schedule:

```bash
# Run batch inference manually
databricks bundle run batch_inference

# Or let it run on schedule (configured in resources/batch_job.yml)
```

## Troubleshooting

### Model not registering

Check that:
1. MLflow experiment is accessible
2. You have MANAGE permissions on the model
3. Unity Catalog model registry is enabled

### Endpoint not starting

Verify:
1. Model artifact is valid
2. You have endpoint creation permissions
3. Workspace has Model Serving enabled

### Inference errors

Check endpoint logs:
```bash
databricks serving-endpoints logs your_model_endpoint
```
