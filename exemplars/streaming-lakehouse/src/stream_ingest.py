# Databricks notebook source
# MAGIC %md
# MAGIC # Streaming Ingestion
# MAGIC Auto Loader-based streaming ingestion with exactly-once semantics.

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    TimestampType,
    DoubleType,
)

# COMMAND ----------

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
source_path = dbutils.widgets.get("source_path")
checkpoint_path = dbutils.widgets.get("checkpoint_path")

# Set catalog context
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Source Schema (optional - can use schema inference)

# For better performance, define schema explicitly
# Comment out to use Auto Loader schema inference
event_schema = StructType([
    StructField("event_id", StringType(), False),
    StructField("event_type", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("event_time", TimestampType(), True),
    StructField("properties", StringType(), True),
    StructField("value", DoubleType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest with Auto Loader

raw_stream = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", f"{checkpoint_path}/schema")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaHints", "event_time timestamp, value double")
    # Uncomment to use explicit schema instead of inference:
    # .schema(event_schema)
    .load(source_path)
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.input_file_name())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Raw Events Table

raw_query = (
    raw_stream
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{checkpoint_path}/raw_events")
    .option("mergeSchema", "true")
    .table(f"{catalog}.{schema}.raw_events")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process and Write to Processed Events Table

processed_stream = (
    spark.readStream
    .table(f"{catalog}.{schema}.raw_events")
    .withWatermark("event_time", "5 minutes")
    .filter(F.col("event_id").isNotNull())
    .filter(F.col("event_type").isNotNull())
    .withColumn("event_date", F.to_date("event_time"))
    .withColumn("event_hour", F.hour("event_time"))
    .withColumn("_processed_at", F.current_timestamp())
)

processed_query = (
    processed_stream
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{checkpoint_path}/processed_events")
    .table(f"{catalog}.{schema}.processed_events")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregated Metrics (Windowed)

metrics_stream = (
    spark.readStream
    .table(f"{catalog}.{schema}.processed_events")
    .withWatermark("event_time", "10 minutes")
    .groupBy(
        F.window("event_time", "5 minutes"),
        "event_type"
    )
    .agg(
        F.count("event_id").alias("event_count"),
        F.countDistinct("user_id").alias("unique_users"),
        F.avg("value").alias("avg_value"),
        F.sum("value").alias("total_value"),
    )
    .select(
        F.col("window.start").alias("window_start"),
        F.col("window.end").alias("window_end"),
        "event_type",
        "event_count",
        "unique_users",
        "avg_value",
        "total_value",
    )
)

metrics_query = (
    metrics_stream
    .writeStream
    .format("delta")
    .outputMode("update")
    .option("checkpointLocation", f"{checkpoint_path}/event_metrics")
    .table(f"{catalog}.{schema}.event_metrics")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Keep notebook running for continuous mode

# Wait for any stream to terminate (they shouldn't in continuous mode)
spark.streams.awaitAnyTermination()
