# Databricks notebook source
# MAGIC %md
# MAGIC # Model Validation
# MAGIC
# MAGIC Validate the trained model before deployment:
# MAGIC 1. Register model to Unity Catalog
# MAGIC 2. Compare against current Champion (if exists)
# MAGIC 3. Test model loading and inference
# MAGIC 4. Set Challenger alias for the new model

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import time

# COMMAND ----------

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

# Get values from training task
run_id = dbutils.jobs.taskValues.get(taskKey="train", key="run_id")
test_mae = dbutils.jobs.taskValues.get(taskKey="train", key="test_mae")

uc_model_name = f"{catalog}.{schema}.{model_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model to Unity Catalog

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# Register the model
model_uri = f"runs:/{run_id}/model"

print(f"Registering model from run: {run_id}")
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=uc_model_name
)

new_version = registered_model.version
print(f"Registered model: {uc_model_name} version {new_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare with Current Champion

# COMMAND ----------

# Check if Champion exists
champion_mae = None
try:
    champion_version = client.get_model_version_by_alias(uc_model_name, "Champion")
    champion_run_id = champion_version.run_id

    # Get Champion's MAE from MLflow run
    champion_run = client.get_run(champion_run_id)
    champion_mae = champion_run.data.metrics.get("test_mae")

    print(f"Current Champion: version {champion_version.version}")
    print(f"Champion MAE: ${champion_mae:.2f}")
    print(f"Challenger MAE: ${test_mae:.2f}")

    if test_mae < champion_mae:
        print("New model OUTPERFORMS Champion!")
        is_better = True
    else:
        print("New model does not outperform Champion")
        is_better = False

except mlflow.exceptions.MlflowException:
    print("No Champion exists yet - this will be the first!")
    is_better = True

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Model Loading and Inference
# MAGIC
# MAGIC This validates the model can be loaded and used for predictions.
# MAGIC Handles cold start by retrying with backoff.

# COMMAND ----------

def load_model_with_retry(model_uri: str, max_retries: int = 5) -> object:
    """Load model with retry logic for cold start handling.

    Args:
        model_uri: URI of the model to load
        max_retries: Maximum number of retry attempts

    Returns:
        Loaded model object
    """
    for attempt in range(max_retries):
        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

# Test loading the registered model
test_model_uri = f"models:/{uc_model_name}/{new_version}"
print(f"Testing model load from: {test_model_uri}")

model = load_model_with_retry(test_model_uri)
print("Model loaded successfully!")

# COMMAND ----------

# Test inference with sample data
test_input = pd.DataFrame({
    "trip_distance": [2.5, 5.0, 10.0],
    "pickup_hour": [8, 14, 22],
    "pickup_dayofweek": [2, 5, 7],
    "pickup_month": [1, 6, 12],
    "is_weekend": [0, 0, 1],
    "is_rush_hour": [1, 0, 0]
})

predictions = model.predict(test_input)
print("Test predictions:")
for i, (dist, pred) in enumerate(zip(test_input["trip_distance"], predictions)):
    print(f"  Trip {dist} miles: ${pred:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Model Alias

# COMMAND ----------

# Set Challenger alias on the new model
client.set_registered_model_alias(
    name=uc_model_name,
    alias="Challenger",
    version=new_version
)
print(f"Set 'Challenger' alias on version {new_version}")

# Update model version description
client.update_model_version(
    name=uc_model_name,
    version=new_version,
    description=f"Trained from run {run_id}. Test MAE: ${test_mae:.2f}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pass Results to Deploy Task

# COMMAND ----------

# Pass validation results to deploy task
dbutils.jobs.taskValues.set(key="model_version", value=int(new_version))
dbutils.jobs.taskValues.set(key="is_better", value=is_better)
dbutils.jobs.taskValues.set(key="test_mae", value=test_mae)

print(f"\nValidation complete!")
print(f"Model version: {new_version}")
print(f"Ready for deployment: {is_better}")
