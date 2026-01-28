# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering
# MAGIC Create and populate the feature table for model training.

from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient

# COMMAND ----------

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Raw Data

# Replace with your actual data source
raw_data = spark.table(f"{catalog}.{schema}.raw_training_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Transformations

def create_features(df):
    """Create features from raw data.

    Args:
        df: Raw input DataFrame

    Returns:
        DataFrame with engineered features
    """
    return (
        df
        # Example feature transformations - customize for your use case
        .withColumn("feature_ratio", F.col("value_a") / F.col("value_b"))
        .withColumn("feature_log", F.log1p(F.col("amount")))
        .withColumn("feature_binned",
            F.when(F.col("amount") < 100, "low")
            .when(F.col("amount") < 1000, "medium")
            .otherwise("high")
        )
        .withColumn("feature_date_dayofweek", F.dayofweek("event_date"))
        .withColumn("feature_date_month", F.month("event_date"))
        # Add more features as needed
    )


features_df = create_features(raw_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feature Table

fe_client = FeatureEngineeringClient()

# Create or update the feature table
feature_table_name = f"{catalog}.{schema}.feature_table"

# Select only the features and primary key
feature_columns = [
    "id",  # Primary key
    "feature_ratio",
    "feature_log",
    "feature_binned",
    "feature_date_dayofweek",
    "feature_date_month",
]

features_final = features_df.select(feature_columns)

# Write to feature table (creates if not exists)
features_final.write.mode("overwrite").saveAsTable(feature_table_name)

print(f"Feature table created/updated: {feature_table_name}")
print(f"Row count: {features_final.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Training Dataset with Labels

# Join features with labels for training
training_data = (
    features_df
    .select(
        "id",
        "feature_ratio",
        "feature_log",
        "feature_binned",
        "feature_date_dayofweek",
        "feature_date_month",
        "label"  # Your target variable
    )
)

training_table_name = f"{catalog}.{schema}.training_dataset"
training_data.write.mode("overwrite").saveAsTable(training_table_name)

print(f"Training dataset created: {training_table_name}")
