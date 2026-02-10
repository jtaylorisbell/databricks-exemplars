# Databricks notebook source
# MAGIC %md
# MAGIC # Model Monitoring
# MAGIC
# MAGIC Monitor model performance using **Lakehouse Monitoring** on the AI Gateway Inference Table.
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Reads raw request/response payloads from the inference table
# MAGIC 2. Unpacks JSON into a structured Delta table
# MAGIC 3. Joins with ground truth (actual fare amounts) from the feature table
# MAGIC 4. Creates or refreshes a Lakehouse Monitor for automated drift detection and quality tracking

# COMMAND ----------

import json

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from pyspark.sql import functions as F, types as T

# COMMAND ----------

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

endpoint_name = f"{schema}_{model_name}_endpoint"
inference_table = f"{catalog}.{schema}.`{endpoint_name}_payload`"
processed_table = f"{catalog}.{schema}.{model_name}_inference_processed"
feature_table = f"{catalog}.{schema}.feature_table"

print(f"Endpoint: {endpoint_name}")
print(f"Inference table: {inference_table}")
print(f"Processed table: {processed_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Endpoint

# COMMAND ----------

w = WorkspaceClient()
endpoint = w.serving_endpoints.get(endpoint_name)

served_entity = endpoint.config.served_entities[0]
print(f"Served model: {served_entity.entity_name} v{served_entity.entity_version}")
print(f"Endpoint state: {endpoint.state.ready}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Request/Response Schema

# COMMAND ----------

# Schema for our fare prediction model's inputs and outputs
request_fields = [
    T.StructField("trip_distance", T.FloatType()),
    T.StructField("pickup_hour", T.IntegerType()),
    T.StructField("pickup_dayofweek", T.IntegerType()),
    T.StructField("pickup_month", T.IntegerType()),
    T.StructField("is_weekend", T.IntegerType()),
    T.StructField("is_rush_hour", T.IntegerType()),
]
response_field = T.StructField("predicted_fare", T.FloatType())

PREDICTION_COL = "predicted_fare"
LABEL_COL = "fare_amount"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unpack Inference Table Payloads

# COMMAND ----------

# JSON consolidation UDF — normalizes multiple Model Serving request formats
# into a common record-oriented structure for consistent downstream processing.
def convert_to_record_json(json_str: str) -> str:
    """Convert from any accepted JSON format into record-oriented format."""
    try:
        request = json.loads(json_str)
    except json.JSONDecodeError:
        return json_str

    output = []
    if isinstance(request, dict):
        keys = set(request.keys())
        if "dataframe_records" in keys:
            output.extend(request["dataframe_records"])
        elif "dataframe_split" in keys:
            split = request["dataframe_split"]
            output.extend(
                dict(zip(split["columns"], vals)) for vals in split["data"]
            )
        elif "instances" in keys:
            output.extend(request["instances"])
        elif "inputs" in keys:
            output.extend(request["inputs"])
        elif "predictions" in keys:
            output.extend({PREDICTION_COL: p} for p in request["predictions"])
    return json.dumps(output)


@F.pandas_udf(T.StringType())
def consolidate_json(json_strs: pd.Series) -> pd.Series:
    return json_strs.apply(convert_to_record_json)

# COMMAND ----------

# Read raw inference table and filter to successful requests
raw_df = spark.table(inference_table).filter(F.col("status_code") == "200")

record_count = raw_df.count()
print(f"Successful inference records: {record_count}")

if record_count == 0:
    print("No inference records to process. Exiting.")
    dbutils.notebook.exit("NO_DATA")

# COMMAND ----------

# Consolidate JSON formats and parse into structured columns
request_schema = T.ArrayType(T.StructType(request_fields))
response_schema = T.ArrayType(T.StructType([response_field]))

unpacked_df = (
    raw_df
    .withColumn("request", consolidate_json(F.col("request")))
    .withColumn("response", consolidate_json(F.col("response")))
    .withColumn("request", F.from_json(F.col("request"), request_schema))
    .withColumn("response", F.from_json(F.col("response"), response_schema))
)

# Explode batched requests into individual rows and flatten
exploded_df = (
    unpacked_df
    .withColumn("req_resp", F.arrays_zip(F.col("request"), F.col("response")))
    .withColumn("req_resp", F.explode(F.col("req_resp")))
    .select(
        F.col("request_time").cast(T.TimestampType()).alias("timestamp"),
        F.col("request_time").cast("date").alias("date"),
        F.coalesce(F.col("served_entity_id"), F.lit("unknown")).alias("model_id"),
        F.col("req_resp.request.*"),
        F.col("req_resp.response.*"),
        F.expr("uuid()").alias("example_id"),
    )
)

print(f"Exploded to {exploded_df.count()} individual predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join with Ground Truth

# COMMAND ----------

# Join with feature table to attach actual fare amounts for quality metrics.
# Uses a left join so predictions without matching actuals are still monitored.
features_df = spark.table(feature_table)

feature_join_cols = [
    "trip_distance", "pickup_hour", "pickup_dayofweek",
    "pickup_month", "is_weekend", "is_rush_hour",
]

processed_df = (
    exploded_df
    .join(
        features_df.select(*feature_join_cols, LABEL_COL),
        on=feature_join_cols,
        how="left",
    )
)

matched = processed_df.filter(F.col(LABEL_COL).isNotNull()).count()
total = processed_df.count()
print(f"Matched with ground truth: {matched} / {total}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Processed Table

# COMMAND ----------

(
    processed_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(processed_table)
)

# Enable Change Data Feed — required by Lakehouse Monitoring
spark.sql(
    f"ALTER TABLE {processed_table} "
    f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

print(f"Processed table written: {processed_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or Refresh Lakehouse Monitor
# MAGIC
# MAGIC Lakehouse Monitoring automatically tracks:
# MAGIC - **Data drift**: detects shifts in input feature distributions
# MAGIC - **Model quality**: regression metrics (MAE, RMSE, R2) when ground truth is available
# MAGIC - **Statistical profiles**: summary statistics over configurable time windows
# MAGIC
# MAGIC Results are available in the **Quality** tab of the table in Catalog Explorer.

# COMMAND ----------

monitor_exists = False
try:
    w.quality_monitors.get(table_name=processed_table)
    monitor_exists = True
    print(f"Monitor already exists for {processed_table}")
except Exception:
    print(f"Creating new monitor for {processed_table}")

# COMMAND ----------

if monitor_exists:
    w.quality_monitors.run_refresh(table_name=processed_table)
    print("Monitor refresh triggered")
else:
    w.quality_monitors.create(
        table_name=processed_table,
        assets_dir=f"/Shared/lakehouse_monitoring/{endpoint_name}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
            prediction_col=PREDICTION_COL,
            label_col=LABEL_COL,
            timestamp_col="timestamp",
            model_id_col="model_id",
            granularities=["1 day"],
        ),
    )
    print(f"Monitor created for {processed_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "=" * 50)
print("MONITORING SUMMARY")
print("=" * 50)
print(f"Inference records processed: {total}")
print(f"Ground truth matched: {matched}")
print(f"Processed table: {processed_table}")
print(f"Monitor status: {'refreshed' if monitor_exists else 'created'}")
print(f"\nView dashboards: Catalog Explorer → {processed_table} → Quality tab")
print("=" * 50)
