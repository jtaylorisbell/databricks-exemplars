# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training
# MAGIC
# MAGIC Train a fare prediction model using NYC Taxi data with MLflow experiment tracking.
# MAGIC Uses `samples.nyctaxi.trips` as source data.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# COMMAND ----------

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.mlflow_artifacts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Prepare NYC Taxi Data

# COMMAND ----------

# Load nyctaxi trips data
raw_df = spark.table("samples.nyctaxi.trips")

# Feature engineering
features_df = (
    raw_df
    .filter(F.col("fare_amount") > 0)
    .filter(F.col("fare_amount") < 200)  # Remove outliers
    .filter(F.col("trip_distance") > 0)
    .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    .withColumn("pickup_dayofweek", F.dayofweek("tpep_pickup_datetime"))
    .withColumn("pickup_month", F.month("tpep_pickup_datetime"))
    .withColumn("is_weekend",
        F.when(F.dayofweek("tpep_pickup_datetime").isin([1, 7]), 1).otherwise(0))
    .withColumn("is_rush_hour",
        F.when(
            (F.hour("tpep_pickup_datetime").between(7, 9)) |
            (F.hour("tpep_pickup_datetime").between(16, 19)), 1
        ).otherwise(0))
    .select(
        "trip_distance",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "is_rush_hour",
        "fare_amount"
    )
)

print(f"Total records after filtering: {features_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Feature Table

# COMMAND ----------

# Save feature table for batch inference
feature_table_name = f"{catalog}.{schema}.feature_table"
features_df.write.mode("overwrite").saveAsTable(feature_table_name)
print(f"Feature table saved: {feature_table_name}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Training Data

# COMMAND ----------

# Convert to pandas for sklearn
pdf = features_df.toPandas()

feature_columns = [
    "trip_distance",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "is_rush_hour"
]

X = pdf[feature_columns]
y = pdf["fare_amount"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train with MLflow Tracking

# COMMAND ----------

# Initialize MLflow with artifact storage in Unity Catalog Volume
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

experiment_path = f"/Shared/{catalog}_{schema}_{model_name}"
artifact_location = f"dbfs:/Volumes/{catalog}/{schema}/mlflow_artifacts"

if mlflow.get_experiment_by_name(experiment_path) is None:
    mlflow.create_experiment(name=experiment_path, artifact_location=artifact_location)
mlflow.set_experiment(experiment_path)

# Hyperparameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
}

with mlflow.start_run(run_name="fare_prediction_training") as run:
    # Log dataset info
    mlflow.log_param("dataset", "samples.nyctaxi.trips")
    mlflow.log_param("feature_columns", feature_columns)
    mlflow.log_param("training_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))

    # Hyperparameter search
    base_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)

    # Evaluate on test set
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_r2", r2)

    print(f"Test MAE: ${mae:.2f}")
    print(f"Test RMSE: ${rmse:.2f}")
    print(f"Test R2: {r2:.4f}")

    # Infer signature and log model
    signature = infer_signature(X_train, best_model.predict(X_train))

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
    )

    # Pass run ID to next task
    run_id = run.info.run_id
    dbutils.jobs.taskValues.set(key="run_id", value=run_id)
    dbutils.jobs.taskValues.set(key="test_mae", value=mae)

    print(f"\nMLflow Run ID: {run_id}")
    print(f"Best parameters: {grid_search.best_params_}")
