# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Inference
# MAGIC
# MAGIC Score new data using `ai_query()` against the deployed serving endpoint.
# MAGIC This avoids loading the model onto the driver and instead routes predictions
# MAGIC through the Model Serving endpoint, which also logs requests to the inference table.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F
from pyspark.sql.functions import expr

# COMMAND ----------

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

feature_table = f"{catalog}.{schema}.feature_table"
predictions_table = f"{catalog}.{schema}.predictions"
endpoint_name = f"{schema}_{model_name}_endpoint"

print(f"Endpoint: {endpoint_name}")
print(f"Feature table: {feature_table}")
print(f"Predictions table: {predictions_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Current Model Version

# COMMAND ----------

w = WorkspaceClient()
endpoint = w.serving_endpoints.get(endpoint_name)
current_model_version = endpoint.config.served_entities[0].entity_version
print(f"Serving model version: {current_model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Features for Scoring

# COMMAND ----------

features_df = spark.table(feature_table)
record_count = features_df.count()
print(f"Records to score: {record_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Predictions via ai_query()

# COMMAND ----------

predictions_df = (
    features_df
    .withColumn(
        "predicted_fare",
        expr(f"""
            ai_query(
                endpoint => '{endpoint_name}',
                request => named_struct(
                    'trip_distance', trip_distance,
                    'pickup_hour', pickup_hour,
                    'pickup_dayofweek', pickup_dayofweek,
                    'pickup_month', pickup_month,
                    'is_weekend', is_weekend,
                    'is_rush_hour', is_rush_hour
                ),
                returnType => 'FLOAT'
            )
        """)
    )
    .withColumn("prediction_timestamp", F.current_timestamp())
    .withColumn("model_version", F.lit(current_model_version))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Predictions
# MAGIC
# MAGIC This predictions table is a convenience layer for downstream consumers.
# MAGIC The raw request/response data is also logged automatically to the AI Gateway
# MAGIC inference table, which the monitoring job uses for drift detection and quality tracking.

# COMMAND ----------

(
    predictions_df
    .select(
        "trip_distance",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "is_rush_hour",
        "fare_amount",
        "predicted_fare",
        "prediction_timestamp",
        "model_version",
    )
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(predictions_table)
)

print(f"Predictions written to: {predictions_table}")
print(f"Records scored: {record_count}")
print(f"Model version: {current_model_version}")
