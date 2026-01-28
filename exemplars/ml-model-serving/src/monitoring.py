# Databricks notebook source
# MAGIC %md
# MAGIC # Model Monitoring
# MAGIC
# MAGIC Monitor model performance and detect drift indicators:
# MAGIC 1. Prediction distribution drift
# MAGIC 2. Feature distribution drift
# MAGIC 3. Model accuracy degradation
# MAGIC
# MAGIC Writes metrics to a monitoring table for alerting and dashboards.

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime, timedelta

# COMMAND ----------

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

baseline_table = f"{catalog}.{schema}.training_baseline"
predictions_table = f"{catalog}.{schema}.predictions"
monitoring_table = f"{catalog}.{schema}.model_monitoring"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Baseline Statistics

# COMMAND ----------

# Load training baseline stats
baseline_df = spark.table(baseline_table)

# Parse baseline into a usable format
baseline_stats = {}
for row in baseline_df.collect():
    summary_type = row["summary"]
    baseline_stats[summary_type] = {
        "trip_distance": float(row["trip_distance"]) if row["trip_distance"] else None,
        "pickup_hour": float(row["pickup_hour"]) if row["pickup_hour"] else None,
        "pickup_dayofweek": float(row["pickup_dayofweek"]) if row["pickup_dayofweek"] else None,
        "fare_amount": float(row["fare_amount"]) if row["fare_amount"] else None,
    }

print("Training baseline loaded")
print(f"  Baseline fare mean: ${baseline_stats['mean']['fare_amount']:.2f}")
print(f"  Baseline fare stddev: ${baseline_stats['stddev']['fare_amount']:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Recent Predictions

# COMMAND ----------

# Get predictions from last 24 hours
cutoff_time = datetime.now() - timedelta(hours=24)

recent_predictions = (
    spark.table(predictions_table)
    .filter(F.col("prediction_timestamp") >= cutoff_time)
)

recent_count = recent_predictions.count()
print(f"Recent predictions (last 24h): {recent_count}")

if recent_count == 0:
    print("No recent predictions to analyze. Exiting.")
    dbutils.notebook.exit("NO_DATA")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate Drift Metrics

# COMMAND ----------

# Calculate current distribution statistics
current_stats = (
    recent_predictions
    .agg(
        F.mean("predicted_fare").alias("pred_mean"),
        F.stddev("predicted_fare").alias("pred_stddev"),
        F.mean("trip_distance").alias("distance_mean"),
        F.stddev("trip_distance").alias("distance_stddev"),
        F.mean(F.abs(F.col("predicted_fare") - F.col("fare_amount"))).alias("mae"),
        F.count("*").alias("record_count")
    )
    .first()
)

# Calculate drift indicators
baseline_fare_mean = baseline_stats["mean"]["fare_amount"]
baseline_fare_std = baseline_stats["stddev"]["fare_amount"]
baseline_distance_mean = baseline_stats["mean"]["trip_distance"]
baseline_distance_std = baseline_stats["stddev"]["trip_distance"]

# Prediction drift: how far is current prediction mean from baseline target mean
pred_drift = abs(current_stats.pred_mean - baseline_fare_mean) / baseline_fare_std

# Feature drift: how far is current feature mean from baseline
feature_drift = abs(current_stats.distance_mean - baseline_distance_mean) / baseline_distance_std

print(f"\nDrift Analysis:")
print(f"  Prediction mean: ${current_stats.pred_mean:.2f} (baseline: ${baseline_fare_mean:.2f})")
print(f"  Prediction drift (z-score): {pred_drift:.2f}")
print(f"  Feature drift (z-score): {feature_drift:.2f}")
print(f"  Current MAE: ${current_stats.mae:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Determine Alerts

# COMMAND ----------

# Thresholds for drift detection
DRIFT_THRESHOLD = 2.0  # Z-score threshold
MAE_THRESHOLD = 5.0    # Maximum acceptable MAE

alerts = []

if pred_drift > DRIFT_THRESHOLD:
    alerts.append(f"PREDICTION_DRIFT: z-score {pred_drift:.2f} exceeds threshold {DRIFT_THRESHOLD}")

if feature_drift > DRIFT_THRESHOLD:
    alerts.append(f"FEATURE_DRIFT: z-score {feature_drift:.2f} exceeds threshold {DRIFT_THRESHOLD}")

if current_stats.mae > MAE_THRESHOLD:
    alerts.append(f"ACCURACY_DEGRADATION: MAE ${current_stats.mae:.2f} exceeds threshold ${MAE_THRESHOLD}")

should_retrain = len(alerts) > 0

if should_retrain:
    print("\n⚠️  ALERTS TRIGGERED:")
    for alert in alerts:
        print(f"  - {alert}")
    print("\n  Recommendation: Consider retraining the model")
else:
    print("\n✓ No drift detected. Model performing within acceptable bounds.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Monitoring Metrics

# COMMAND ----------

# Create monitoring record
monitoring_record = spark.createDataFrame([{
    "timestamp": datetime.now(),
    "model_name": f"{catalog}.{schema}.{model_name}",
    "record_count": current_stats.record_count,
    "prediction_mean": float(current_stats.pred_mean),
    "prediction_stddev": float(current_stats.pred_stddev),
    "baseline_mean": baseline_fare_mean,
    "baseline_stddev": baseline_fare_std,
    "prediction_drift_zscore": float(pred_drift),
    "feature_drift_zscore": float(feature_drift),
    "mae": float(current_stats.mae),
    "drift_threshold": DRIFT_THRESHOLD,
    "mae_threshold": MAE_THRESHOLD,
    "alerts": ",".join(alerts) if alerts else None,
    "should_retrain": should_retrain
}])

# Append to monitoring table
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

print("\n" + "="*50)
print("MONITORING SUMMARY")
print("="*50)
print(f"Period: Last 24 hours")
print(f"Records analyzed: {current_stats.record_count}")
print(f"Prediction drift: {pred_drift:.2f} (threshold: {DRIFT_THRESHOLD})")
print(f"Feature drift: {feature_drift:.2f} (threshold: {DRIFT_THRESHOLD})")
print(f"MAE: ${current_stats.mae:.2f} (threshold: ${MAE_THRESHOLD})")
print(f"Retrain recommended: {should_retrain}")
print("="*50)
