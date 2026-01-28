# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Inference
# MAGIC Score new data using the registered model.

import mlflow
from pyspark.sql import functions as F

# COMMAND ----------

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model from Unity Catalog

# Full model path
uc_model_name = f"{catalog}.{schema}.{model_name}"

# Load model using alias (champion) or specific version
model_uri = f"models:/{uc_model_name}@champion"
model = mlflow.pyfunc.load_model(model_uri)

print(f"Loaded model: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data to Score

# Load feature data (customize based on your source)
scoring_data = spark.table(f"{catalog}.{schema}.feature_table")

# Convert to pandas for sklearn model
scoring_pdf = scoring_data.toPandas()

# Define feature columns (must match training)
feature_columns = [
    "feature_ratio",
    "feature_log",
    "feature_binned_encoded",
    "feature_date_dayofweek",
    "feature_date_month",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Predictions

# Score
predictions = model.predict(scoring_pdf[feature_columns])

# Add predictions to dataframe
scoring_pdf["prediction"] = predictions
scoring_pdf["scored_at"] = F.current_timestamp()

# Convert back to Spark
predictions_df = spark.createDataFrame(scoring_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Predictions

# Append predictions with timestamp
(
    predictions_df
    .withColumn("scored_at", F.current_timestamp())
    .write
    .mode("append")
    .saveAsTable(f"{catalog}.{schema}.predictions")
)

print(f"Predictions written to {catalog}.{schema}.predictions")
print(f"Rows scored: {predictions_df.count()}")
