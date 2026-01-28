# Databricks notebook source
# MAGIC %md
# MAGIC # Register Model
# MAGIC Register the trained model to Unity Catalog Model Registry.

import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

# Get run ID from training task
run_id = dbutils.jobs.taskValues.get(taskKey="train", key="best_run_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model to Unity Catalog

# Full model path in Unity Catalog
uc_model_name = f"{catalog}.{schema}.{model_name}"

# Set registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Register the model
model_uri = f"runs:/{run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=uc_model_name
)

print(f"Model registered: {uc_model_name}")
print(f"Version: {registered_model.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Model Description

client = MlflowClient()

# Update model description
client.update_registered_model(
    name=uc_model_name,
    description="Production model trained with RandomForest classifier"
)

# Update version description
client.update_model_version(
    name=uc_model_name,
    version=registered_model.version,
    description=f"Trained from run {run_id}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Model Alias (Optional)

# Set alias for easy reference
client.set_registered_model_alias(
    name=uc_model_name,
    alias="champion",
    version=registered_model.version
)

print(f"Set 'champion' alias to version {registered_model.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output for Downstream Tasks

# Make version available to subsequent tasks
dbutils.jobs.taskValues.set(key="model_version", value=registered_model.version)
print(f"\nModel ready for serving: {uc_model_name} version {registered_model.version}")
