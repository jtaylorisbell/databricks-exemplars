# Databricks notebook source
# MAGIC %md
# MAGIC # Model Monitoring
# MAGIC
# MAGIC Monitor model performance using the AI Gateway Inference Table.
# MAGIC The inference table automatically logs all requests and responses from the
# MAGIC serving endpoint, providing a built-in audit trail for monitoring.
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Reads recent requests/responses from the inference table
# MAGIC 2. Joins with the feature table to get actual fare amounts
# MAGIC 3. Computes accuracy (MAE) and prediction distribution stats
# MAGIC 4. Writes monitoring records for alerting and dashboards

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, ArrayType, StructType, StructField, IntegerType
from datetime import datetime, timedelta

# COMMAND ----------

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

endpoint_name = f"{schema}_{model_name}_endpoint"
inference_table = f"{catalog}.{schema}.`{endpoint_name}_payload`"
feature_table = f"{catalog}.{schema}.feature_table"
monitoring_table = f"{catalog}.{schema}.model_monitoring"

print(f"Inference table: {inference_table}")
print(f"Feature table: {feature_table}")
print(f"Monitoring table: {monitoring_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Recent Inference Requests

# COMMAND ----------

# Get inference records from the last 24 hours
cutoff_time = datetime.now() - timedelta(hours=24)

raw_payloads = (
    spark.table(inference_table)
    .filter(F.col("request_time") >= F.lit(cutoff_time).cast("timestamp"))
)

recent_count = raw_payloads.count()
print(f"Recent inference records (last 24h): {recent_count}")

if recent_count == 0:
    print("No recent inference records to analyze. Exiting.")
    dbutils.notebook.exit("NO_DATA")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse Requests and Responses

# COMMAND ----------

# The inference table stores request/response as JSON strings.
# For custom model serving endpoints:
#   request:  {"dataframe_records": [{"col1": val, ...}]}
#   response: {"predictions": [val, ...]}

request_schema = "struct<dataframe_records:array<struct<trip_distance:float, pickup_hour:int, pickup_dayofweek:int, pickup_month:int, is_weekend:int, is_rush_hour:int>>>"
response_schema = "struct<predictions:array<float>>"

parsed_df = (
    raw_payloads
    .withColumn("req_parsed", F.from_json(F.col("request"), request_schema))
    .withColumn("resp_parsed", F.from_json(F.col("response"), response_schema))
    # Explode the arrays to get one row per record
    .select(
        F.posexplode(F.col("req_parsed.dataframe_records")).alias("pos", "record"),
        F.col("resp_parsed.predictions").alias("predictions"),
    )
    .select(
        F.col("record.trip_distance").alias("trip_distance"),
        F.col("record.pickup_hour").alias("pickup_hour"),
        F.col("record.pickup_dayofweek").alias("pickup_dayofweek"),
        F.col("record.pickup_month").alias("pickup_month"),
        F.col("record.is_weekend").alias("is_weekend"),
        F.col("record.is_rush_hour").alias("is_rush_hour"),
        F.col("predictions").getItem(F.col("pos")).alias("predicted_fare"),
    )
)

parsed_count = parsed_df.count()
print(f"Parsed {parsed_count} inference records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join with Actuals

# COMMAND ----------

# Join with feature table to get actual fare amounts for accuracy calculation
features_df = spark.table(feature_table)

joined_df = (
    parsed_df
    .join(
        features_df.select(
            "trip_distance", "pickup_hour", "pickup_dayofweek",
            "pickup_month", "is_weekend", "is_rush_hour", "fare_amount"
        ),
        on=["trip_distance", "pickup_hour", "pickup_dayofweek",
            "pickup_month", "is_weekend", "is_rush_hour"],
        how="inner",
    )
)

matched_count = joined_df.count()
print(f"Matched with actuals: {matched_count} of {parsed_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate Monitoring Metrics

# COMMAND ----------

# Prediction distribution stats (from all inference records)
pred_stats = (
    parsed_df
    .agg(
        F.mean("predicted_fare").alias("pred_mean"),
        F.stddev("predicted_fare").alias("pred_stddev"),
        F.count("*").alias("record_count"),
    )
    .first()
)

# Accuracy metrics (only where we have actuals)
mae = None
if matched_count > 0:
    accuracy_stats = (
        joined_df
        .agg(
            F.mean(F.abs(F.col("predicted_fare") - F.col("fare_amount"))).alias("mae"),
        )
        .first()
    )
    mae = float(accuracy_stats.mae)

print(f"\nPrediction Distribution:")
print(f"  Mean: ${pred_stats.pred_mean:.2f}")
print(f"  Stddev: ${pred_stats.pred_stddev:.2f}")
print(f"  Count: {pred_stats.record_count}")
if mae is not None:
    print(f"  MAE vs actuals: ${mae:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Determine Alerts

# COMMAND ----------

MAE_THRESHOLD = 5.0

alerts = []

if mae is not None and mae > MAE_THRESHOLD:
    alerts.append(f"ACCURACY_DEGRADATION: MAE ${mae:.2f} exceeds threshold ${MAE_THRESHOLD}")

should_retrain = len(alerts) > 0

if should_retrain:
    print("\nALERTS TRIGGERED:")
    for alert in alerts:
        print(f"  - {alert}")
    print("\n  Recommendation: Consider retraining the model")
else:
    print("\nNo alerts. Model performing within acceptable bounds.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Monitoring Metrics

# COMMAND ----------

monitoring_schema = "timestamp timestamp, model_name string, endpoint_name string, record_count int, prediction_mean double, prediction_stddev double, mae double, mae_threshold double, alerts string, should_retrain boolean"

monitoring_record = spark.createDataFrame([{
    "timestamp": datetime.now(),
    "model_name": f"{catalog}.{schema}.{model_name}",
    "endpoint_name": endpoint_name,
    "record_count": int(pred_stats.record_count),
    "prediction_mean": float(pred_stats.pred_mean),
    "prediction_stddev": float(pred_stats.pred_stddev),
    "mae": mae,
    "mae_threshold": MAE_THRESHOLD,
    "alerts": ",".join(alerts) if alerts else None,
    "should_retrain": should_retrain,
}], schema=monitoring_schema)

(
    monitoring_record
    .write
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(monitoring_table)
)

print(f"\nMonitoring metrics written to: {monitoring_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "=" * 50)
print("MONITORING SUMMARY")
print("=" * 50)
print(f"Period: Last 24 hours")
print(f"Inference records: {pred_stats.record_count}")
print(f"Prediction mean: ${pred_stats.pred_mean:.2f}")
print(f"Prediction stddev: ${pred_stats.pred_stddev:.2f}")
if mae is not None:
    print(f"MAE: ${mae:.2f} (threshold: ${MAE_THRESHOLD})")
else:
    print(f"MAE: N/A (no matched actuals)")
print(f"Retrain recommended: {should_retrain}")
print("=" * 50)
