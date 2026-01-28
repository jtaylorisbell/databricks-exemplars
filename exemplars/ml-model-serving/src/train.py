# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training
# MAGIC Train model with MLflow experiment tracking and hyperparameter tuning.

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# COMMAND ----------

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set MLflow Experiment

mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Training Data

training_df = spark.table(f"{catalog}.{schema}.training_dataset").toPandas()

# Define features and target
feature_columns = [
    "feature_ratio",
    "feature_log",
    "feature_date_dayofweek",
    "feature_date_month",
]

# Handle categorical features
le = LabelEncoder()
training_df["feature_binned_encoded"] = le.fit_transform(training_df["feature_binned"])
feature_columns.append("feature_binned_encoded")

X = training_df[feature_columns]
y = training_df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Search

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train with MLflow Tracking

with mlflow.start_run(run_name="random_forest_training") as run:
    # Log parameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("feature_columns", feature_columns)

    # Hyperparameter search
    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("cv_best_score", grid_search.best_score_)

    # Evaluate on test set
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1: {f1:.4f}")

    # Infer model signature
    signature = infer_signature(X_train, best_model.predict(X_train))

    # Log model
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
    )

    # Save run ID for registration step
    run_id = run.info.run_id
    dbutils.jobs.taskValues.set(key="best_run_id", value=run_id)

    print(f"\nMLflow Run ID: {run_id}")
    print(f"Best parameters: {grid_search.best_params_}")
