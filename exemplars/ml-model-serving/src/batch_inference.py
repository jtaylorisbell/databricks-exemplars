# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Inference
# MAGIC
# MAGIC Score new data using `ai_query()` against the deployed serving endpoint.
# MAGIC This avoids loading the model onto the driver and instead routes predictions
# MAGIC through the Model Serving endpoint, which also logs requests to the inference table.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import expr
from datetime import datetime

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
    .withColumn("model_version", F.lit(endpoint_name))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Predictions

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
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(predictions_table)
)

print(f"Predictions written to: {predictions_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Inference Metrics

# COMMAND ----------

error_stats = (
    predictions_df
    .withColumn("prediction_error", F.abs(F.col("predicted_fare") - F.col("fare_amount")))
    .select(
        F.mean("prediction_error").alias("mae"),
        F.stddev("prediction_error").alias("error_stddev"),
        F.count("*").alias("record_count"),
    )
    .first()
)

print(f"\nInference batch metrics:")
print(f"  Records scored: {error_stats.record_count}")
print(f"  MAE vs actual: ${error_stats.mae:.2f}")
print(f"  Error stddev: ${error_stats.error_stddev:.2f}")
