# Databricks notebook source
# MAGIC %md
# MAGIC # Model Deployment
# MAGIC
# MAGIC Deploy the validated model to a serving endpoint.
# MAGIC This task creates or updates a Model Serving endpoint with the new model version.
# MAGIC
# MAGIC **Pattern**: Deploy code, not models. The endpoint is created/updated programmatically,
# MAGIC not defined as a DAB resource.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)
from mlflow.tracking import MlflowClient

# COMMAND ----------

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

# Get values from validate task
model_version = dbutils.jobs.taskValues.get(taskKey="validate", key="model_version")
is_better = dbutils.jobs.taskValues.get(taskKey="validate", key="is_better")
test_mae = dbutils.jobs.taskValues.get(taskKey="validate", key="test_mae")

uc_model_name = f"{catalog}.{schema}.{model_name}"
endpoint_name = f"{schema}_{model_name}_endpoint"

print(f"Model: {uc_model_name}")
print(f"Version: {model_version}")
print(f"Is better than Champion: {is_better}")
print(f"Endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Deployment Criteria

# COMMAND ----------

if not is_better:
    print("Model does not outperform current Champion.")
    print("Skipping endpoint deployment.")
    print("Model remains available as 'Challenger' alias for manual review.")
    dbutils.notebook.exit("SKIPPED - Model not better than Champion")

print("Model approved for deployment!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or Update Serving Endpoint

# COMMAND ----------

w = WorkspaceClient()

# Endpoint configuration
served_entity = ServedEntityInput(
    entity_name=uc_model_name,
    entity_version=str(model_version),
    workload_size="Small",
    scale_to_zero_enabled=True,
)

endpoint_config = EndpointCoreConfigInput(
    name=endpoint_name,
    served_entities=[served_entity],
    traffic_config=TrafficConfig(
        routes=[
            Route(
                served_model_name=f"{model_name}-{model_version}",
                traffic_percentage=100
            )
        ]
    )
)

# AI Gateway config — enables automatic request/response logging to an inference table
ai_gateway_config = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(
        catalog_name=catalog,
        schema_name=schema,
        enabled=True,
    )
)

# COMMAND ----------

# Check if endpoint exists
endpoint_exists = False
try:
    existing = w.serving_endpoints.get(endpoint_name)
    endpoint_exists = True
    print(f"Endpoint '{endpoint_name}' exists - will update")
except Exception:
    print(f"Endpoint '{endpoint_name}' does not exist - will create")

# COMMAND ----------

if endpoint_exists:
    # Update existing endpoint
    print(f"Updating endpoint with model version {model_version}...")
    w.serving_endpoints.update_config_and_wait(
        name=endpoint_name,
        served_entities=[served_entity],
        traffic_config=TrafficConfig(
            routes=[
                Route(
                    served_model_name=f"{model_name}-{model_version}",
                    traffic_percentage=100
                )
            ]
        )
    )
    # Enable inference table logging (update_config_and_wait doesn't accept ai_gateway)
    w.serving_endpoints.put_ai_gateway(
        name=endpoint_name,
        inference_table_config=AiGatewayInferenceTableConfig(
            catalog_name=catalog,
            schema_name=schema,
            enabled=True,
        ),
    )
else:
    # Create new endpoint with inference table logging enabled
    print(f"Creating new endpoint '{endpoint_name}'...")
    w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=endpoint_config,
        ai_gateway=ai_gateway_config,
    )

print(f"Endpoint ready: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Deployment

# COMMAND ----------

# Get endpoint status
endpoint = w.serving_endpoints.get(endpoint_name)
print(f"\nEndpoint Status: {endpoint.state.ready}")
print(f"Endpoint URL: {endpoint.config.served_entities[0].entity_name}")

# Test with sample request
import pandas as pd

test_data = pd.DataFrame({
    "trip_distance": [3.5],
    "pickup_hour": [17],
    "pickup_dayofweek": [3],
    "pickup_month": [6],
    "is_weekend": [0],
    "is_rush_hour": [1]
})

response = w.serving_endpoints.query(
    name=endpoint_name,
    dataframe_records=test_data.to_dict(orient="records")
)

print(f"\nTest prediction: ${response.predictions[0]:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote to Champion
# MAGIC
# MAGIC Champion alias is set **after** the endpoint is verified healthy.
# MAGIC This prevents inconsistent state if endpoint creation fails.

# COMMAND ----------

mlflow_client = MlflowClient()

mlflow_client.set_registered_model_alias(
    name=uc_model_name,
    alias="Champion",
    version=model_version
)
print(f"Promoted version {model_version} to 'Champion'")
print(f"\nDeployment complete!")
