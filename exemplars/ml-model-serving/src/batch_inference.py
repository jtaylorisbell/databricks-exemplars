# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Inference
# MAGIC
# MAGIC Score new data using the Champion model and write predictions to a table.

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F
from datetime import datetime

# COMMAND ----------

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

uc_model_name = f"{catalog}.{schema}.{model_name}"
feature_table = f"{catalog}.{schema}.feature_table"
predictions_table = f"{catalog}.{schema}.predictions"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Champion Model

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Load the Champion model using alias
model_uri = f"models:/{uc_model_name}@Champion"
print(f"Loading model: {model_uri}")

model = mlflow.sklearn.load_model(model_uri)
print("Champion model loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Features for Scoring

# COMMAND ----------

# Load feature data
features_df = spark.table(feature_table)
print(f"Records to score: {features_df.count()}")

# Convert to pandas for sklearn model
features_pdf = features_df.toPandas()

feature_columns = [
    "trip_distance",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "is_rush_hour"
]

X = features_pdf[feature_columns]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Predictions

# COMMAND ----------

# Score data
predictions = model.predict(X)
features_pdf["predicted_fare"] = predictions
features_pdf["prediction_timestamp"] = datetime.now()
features_pdf["model_version"] = model_uri

# Convert back to Spark DataFrame
predictions_df = spark.createDataFrame(features_pdf)

print(f"Generated {len(predictions)} predictions")
print(f"Prediction stats:")
print(f"  Mean: ${predictions.mean():.2f}")
print(f"  Min: ${predictions.min():.2f}")
print(f"  Max: ${predictions.max():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Predictions

# COMMAND ----------

# Append predictions to table (creates if not exists)
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
        "model_version"
    )
    .write
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(predictions_table)
)

print(f"Predictions written to: {predictions_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Inference Metrics

# COMMAND ----------

# Calculate prediction error for logged predictions (where actual is available)
error_stats = (
    predictions_df
    .withColumn("prediction_error", F.abs(F.col("predicted_fare") - F.col("fare_amount")))
    .select(
        F.mean("prediction_error").alias("mae"),
        F.stddev("prediction_error").alias("error_stddev"),
        F.count("*").alias("record_count")
    )
    .first()
)

print(f"\nInference batch metrics:")
print(f"  Records scored: {error_stats.record_count}")
print(f"  MAE vs actual: ${error_stats.mae:.2f}")
print(f"  Error stddev: ${error_stats.error_stddev:.2f}")
