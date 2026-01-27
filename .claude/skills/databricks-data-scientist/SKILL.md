---
name: databricks-data-scientist
description: World-class Databricks data scientist specializing in MLflow 3.0, traditional ML, deep learning, and generative AI. Use this skill for experiment tracking, model training, feature engineering, RAG applications, AI agents, model serving, and ML best practices.
---

You are a world-class data scientist specializing in machine learning on the Databricks platform. You have deep expertise in MLflow 3.0, traditional ML, deep learning, and generative AI.

## Core Technology Stack

- **Experiment Tracking**: MLflow 3.0+
- **Traditional ML**: scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch, TensorFlow, Keras
- **GenAI**: LangChain, ChatModel/ResponsesAgent, Mosaic AI
- **Feature Engineering**: Databricks Feature Store + Unity Catalog
- **Vector Search**: Mosaic AI Vector Search
- **Model Serving**: Mosaic AI Model Serving
- **AutoML**: Databricks AutoML

## MLflow 3.0 Fundamentals

MLflow 3.0 introduces significant changes focused on GenAI workflows. Key concepts:

### LoggedModel (New in MLflow 3)

`LoggedModel` is now a first-class entity that persists throughout the model lifecycle:

```python
import mlflow

# MLflow 3.0: Use 'name' parameter instead of 'artifact_path'
# Runs are no longer required for logging models
with mlflow.start_run():
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="my_classifier",  # NEW: replaces artifact_path
        input_example=X_train[:5],
        registered_model_name="prod.ml.my_classifier"  # Unity Catalog path
    )

# LoggedModel contains metadata, metrics, parameters, and code links
print(model_info.model_uri)
```

### Key MLflow 3.0 Changes from 2.x

| Aspect | MLflow 2.x | MLflow 3.0 |
|--------|------------|------------|
| Model logging | `artifact_path` parameter | `name` parameter |
| Run requirement | Required for logging | Optional |
| Artifact storage | Stored as run artifacts | Dedicated models location |
| Removed features | - | MLflow Recipes, fastai/mleap flavors |

### Experiment Tracking

```python
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/me/my_experiment")

with mlflow.start_run(run_name="xgboost_baseline"):
    # Log parameters
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
    })

    # Train model
    model = train_model(X_train, y_train)

    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc,
    })

    # Infer signature from training data
    signature = infer_signature(X_train, model.predict(X_train))

    # Log model with MLflow 3.0 syntax
    mlflow.sklearn.log_model(
        sk_model=model,
        name="xgboost_model",
        signature=signature,
        input_example=X_train[:5],
    )
```

## MLflow Tracing (GenAI Observability)

MLflow 3.0 provides comprehensive tracing for GenAI applications with 20+ library integrations.

### Automatic Tracing

```python
import mlflow

# Enable automatic tracing for supported libraries
mlflow.langchain.autolog()
mlflow.openai.autolog()

# All LangChain/OpenAI calls are now traced automatically
response = chain.invoke({"question": "What is MLflow?"})
```

### Manual Tracing with Decorators

```python
import mlflow

@mlflow.trace(name="process_query")
def process_query(query: str) -> str:
    """Traced function - creates a span automatically."""
    embedding = get_embedding(query)
    results = search_index(embedding)
    return generate_response(results)

@mlflow.trace(name="generate_response", span_type="LLM")
def generate_response(context: str) -> str:
    """Specify span type for better categorization."""
    return llm.invoke(context)
```

### Manual Tracing with Fluent API

```python
import mlflow

def complex_pipeline(query: str) -> str:
    with mlflow.start_span(name="RAG Pipeline") as root_span:
        root_span.set_inputs({"query": query})

        # Retrieval step
        with mlflow.start_span(name="retrieval") as retrieval_span:
            docs = retriever.get_relevant_documents(query)
            retrieval_span.set_outputs({"num_docs": len(docs)})

        # Generation step
        with mlflow.start_span(name="generation", span_type="LLM") as gen_span:
            response = llm.invoke(format_prompt(query, docs))
            gen_span.set_outputs({"response": response})

        root_span.set_outputs({"final_response": response})
        return response
```

### Viewing Traces

Traces appear in the MLflow UI under the experiment's Traces tab. Use the Trace Comparison view for side-by-side analysis across runs.

## Traditional Machine Learning

### Classification/Regression with scikit-learn

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import pandas as pd

mlflow.set_experiment("/Users/me/classification_experiment")

# Load data from Unity Catalog
df = spark.table("catalog.schema.training_data").toPandas()
X = df.drop("target", axis=1)
y = df["target"]

with mlflow.start_run(run_name="random_forest"):
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42,
    }
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1_weighted")
    mlflow.log_metric("cv_f1_mean", cv_scores.mean())
    mlflow.log_metric("cv_f1_std", cv_scores.std())

    # Final fit
    model.fit(X, y)

    # Log feature importances
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    mlflow.log_table(importance_df, artifact_file="feature_importance.json")

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="random_forest_classifier",
        input_example=X.head(),
    )
```

### XGBoost with Early Stopping

```python
import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

with mlflow.start_run(run_name="xgboost_tuned"):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
        "device": "cuda",  # GPU acceleration
    }
    mlflow.log_params(params)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Log best iteration
    mlflow.log_metric("best_iteration", model.best_iteration)
    mlflow.log_metric("best_auc", model.best_score)

    mlflow.xgboost.log_model(
        xgb_model=model,
        name="xgboost_classifier",
        input_example=X_train[:5],
    )
```

### Hyperparameter Tuning with Optuna

```python
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    model = XGBClassifier(**params, eval_metric="auc", early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return model.best_score  # Optuna maximizes by default with direction="maximize"

with mlflow.start_run(run_name="optuna_tuning"):
    # MLflow callback logs each trial as a nested run
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="auc",
        create_experiment=False,
        mlflow_kwargs={"nested": True},
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=50, callbacks=[mlflow_callback])

    # Log best parameters
    mlflow.log_params({"best_" + k: v for k, v in study.best_params.items()})
    mlflow.log_metric("best_auc", study.best_value)

    # Optuna visualization artifacts
    fig = optuna.visualization.plot_optimization_history(study)
    mlflow.log_figure(fig, "optimization_history.html")

    fig = optuna.visualization.plot_param_importances(study)
    mlflow.log_figure(fig, "param_importances.html")
```

## Deep Learning

### PyTorch with Distributed Training

```python
import mlflow
import torch
from torch.distributed import init_process_group
from pyspark.ml.torch.distributor import TorchDistributor

def train_fn():
    import mlflow
    mlflow.pytorch.autolog()

    model = MyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, step=epoch)

    return model

# Distributed training on Databricks
distributor = TorchDistributor(
    num_processes=4,
    local_mode=False,
    use_gpu=True,
)

with mlflow.start_run(run_name="distributed_pytorch"):
    model = distributor.run(train_fn)

    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="pytorch_model",
        input_example=torch.randn(1, 3, 224, 224),
    )
```

### Hugging Face Transformers

```python
import mlflow
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

mlflow.transformers.autolog()

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

with mlflow.start_run(run_name="distilbert_finetuned"):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    # Log the fine-tuned model
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        name="distilbert_classifier",
        task="text-classification",
    )
```

## Generative AI

### ChatModel for Custom GenAI Applications

```python
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatCompletionResponse, ChatMessage, ChatChoice
import mlflow

class MyAgent(ChatModel):
    def __init__(self):
        self.client = None
        self.config = {}

    def load_context(self, context):
        """Called when model is loaded. Initialize clients here."""
        from databricks.sdk import WorkspaceClient
        self.client = WorkspaceClient()
        self.config = context.model_config

    @mlflow.trace(name="agent_predict")
    def predict(self, context, messages: list[ChatMessage], params=None) -> ChatCompletionResponse:
        """Process chat messages and return response."""
        with mlflow.start_span(name="process_messages") as span:
            span.set_inputs({"messages": [m.content for m in messages]})

            # Your agent logic here
            user_message = messages[-1].content
            response = self._generate_response(user_message)

            span.set_outputs({"response": response})

        return ChatCompletionResponse(
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response),
                )
            ],
            model=self.config.get("model_name", "my-agent"),
        )

    def _generate_response(self, query: str) -> str:
        # Call LLM, tools, etc.
        pass

# Log the agent
model_config = {
    "model_name": "my-custom-agent",
    "temperature": 0.7,
    "max_tokens": 1000,
}

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        python_model=MyAgent(),
        name="my_agent",
        model_config=model_config,
        input_example={"messages": [{"role": "user", "content": "Hello"}]},
    )
```

### RAG with Vector Search

```python
from databricks.vector_search.client import VectorSearchClient
import mlflow

# Create vector search client
vsc = VectorSearchClient()

# Create index with managed embeddings
index = vsc.create_delta_sync_index(
    endpoint_name="my_vs_endpoint",
    index_name="catalog.schema.my_index",
    source_table_name="catalog.schema.documents",
    primary_key="doc_id",
    pipeline_type="TRIGGERED",
    embedding_source_column="content",
    embedding_model_endpoint_name="databricks-gte-large-en",  # Managed embedding
)

# Query the index
@mlflow.trace(name="vector_search")
def search_similar(query: str, k: int = 5) -> list[dict]:
    results = index.similarity_search(
        query_text=query,
        columns=["doc_id", "content", "metadata"],
        num_results=k,
    )
    return results.get("result", {}).get("data_array", [])

# RAG chain
@mlflow.trace(name="rag_chain")
def rag_query(question: str) -> str:
    # Retrieve relevant documents
    with mlflow.start_span(name="retrieval") as span:
        docs = search_similar(question, k=5)
        context = "\n\n".join([doc[1] for doc in docs])  # content column
        span.set_outputs({"num_docs": len(docs)})

    # Generate response
    with mlflow.start_span(name="generation", span_type="LLM") as span:
        prompt = f"""Answer based on the context below.

Context:
{context}

Question: {question}

Answer:"""
        response = llm.invoke(prompt)
        span.set_outputs({"response": response})

    return response
```

### LangChain Integration

```python
import mlflow
from langchain_community.chat_models import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Enable automatic tracing
mlflow.langchain.autolog()

# Use Databricks Foundation Model
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

with mlflow.start_run(run_name="langchain_agent"):
    # All chain invocations are traced automatically
    response = chain.invoke({"question": "Explain MLflow in one sentence."})

    # Log the chain as a model
    mlflow.langchain.log_model(
        lc_model=chain,
        name="qa_chain",
        input_example={"question": "What is Databricks?"},
    )
```

## Feature Store

### Creating Feature Tables

```python
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F

fe = FeatureEngineeringClient()

# Create feature table in Unity Catalog
fe.create_table(
    name="catalog.schema.customer_features",
    primary_keys=["customer_id"],
    timestamp_keys=["event_timestamp"],  # For point-in-time lookups
    df=feature_df,
    description="Customer behavioral features",
)

# Update features
fe.write_table(
    name="catalog.schema.customer_features",
    df=new_features_df,
    mode="merge",  # or "overwrite"
)
```

### Training with Feature Lookups

```python
from databricks.feature_engineering import FeatureLookup

# Define feature lookups
feature_lookups = [
    FeatureLookup(
        table_name="catalog.schema.customer_features",
        lookup_key=["customer_id"],
        feature_names=["total_purchases", "avg_order_value", "days_since_last_order"],
        timestamp_lookup_key="event_timestamp",  # Point-in-time correctness
    ),
    FeatureLookup(
        table_name="catalog.schema.product_features",
        lookup_key=["product_id"],
        feature_names=["category", "price_tier"],
    ),
]

# Create training set with automatic feature joining
training_set = fe.create_training_set(
    df=labels_df,  # DataFrame with keys and labels
    feature_lookups=feature_lookups,
    label="purchased",
    exclude_columns=["customer_id", "product_id"],
)

training_df = training_set.load_df()
```

### Log Model with Feature Metadata

```python
with mlflow.start_run():
    model = train_model(training_df)

    # Log model with feature store metadata for automatic lookup at inference
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name="catalog.schema.purchase_predictor",
    )
```

## AutoML

```python
from databricks import automl

# Classification
summary = automl.classify(
    dataset=spark.table("catalog.schema.training_data"),
    target_col="label",
    primary_metric="f1",
    timeout_minutes=60,
    max_trials=100,
)

# Access best model
best_model = summary.best_trial
print(f"Best F1: {best_model.metrics['val_f1_score']}")

# The best model is automatically logged to MLflow
best_model_uri = best_model.model_uri

# Regression
regression_summary = automl.regress(
    dataset=df,
    target_col="price",
    primary_metric="rmse",
)

# Forecasting
forecast_summary = automl.forecast(
    dataset=df,
    target_col="sales",
    time_col="date",
    horizon=30,
    frequency="D",
    primary_metric="smape",
)
```

## Model Serving

### Register Model to Unity Catalog

```python
import mlflow

# Register during logging
with mlflow.start_run():
    mlflow.sklearn.log_model(
        sk_model=model,
        name="my_model",
        registered_model_name="catalog.schema.my_model",  # Unity Catalog path
    )

# Or register existing model
mlflow.register_model(
    model_uri="runs:/abc123/my_model",
    name="catalog.schema.my_model",
)
```

### Deploy to Model Serving

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

w = WorkspaceClient()

# Create serving endpoint
w.serving_endpoints.create(
    name="my-model-endpoint",
    config=EndpointCoreConfigInput(
        served_models=[
            ServedModelInput(
                model_name="catalog.schema.my_model",
                model_version="1",
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ]
    ),
)
```

### Query Endpoint

```python
import requests

# Using the Databricks SDK
w = WorkspaceClient()
response = w.serving_endpoints.query(
    name="my-model-endpoint",
    dataframe_records=[{"feature1": 1.0, "feature2": "value"}],
)

# Or using REST API
endpoint_url = f"https://{workspace_host}/serving-endpoints/my-model-endpoint/invocations"
response = requests.post(
    endpoint_url,
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [{"feature1": 1.0, "feature2": "value"}]},
)
```

## GenAI Evaluation

```python
import mlflow
from mlflow.metrics.genai import relevance, faithfulness

# Evaluate RAG application
eval_data = pd.DataFrame({
    "question": ["What is MLflow?", "How do I log a model?"],
    "context": ["MLflow is an open source platform...", "Use mlflow.log_model()..."],
    "answer": ["MLflow is a platform for ML lifecycle", "Call mlflow.log_model()"],
    "ground_truth": ["MLflow manages the ML lifecycle", "Use the log_model function"],
})

with mlflow.start_run(run_name="rag_evaluation"):
    results = mlflow.evaluate(
        model=rag_chain,
        data=eval_data,
        targets="ground_truth",
        model_type="question-answering",
        evaluators="default",
        extra_metrics=[
            relevance(),
            faithfulness(),
        ],
    )

    print(f"Relevance: {results.metrics['relevance/mean']}")
    print(f"Faithfulness: {results.metrics['faithfulness/mean']}")
```

## Best Practices

### Experiment Organization
- Use descriptive experiment names: `/Users/me/project/model_type`
- Use run names that describe the approach: `xgboost_baseline`, `with_feature_v2`
- Tag runs with metadata: `mlflow.set_tags({"team": "data-science", "dataset": "v2"})`

### Model Logging
- Always include `input_example` for schema inference
- Use `infer_signature()` for explicit signatures
- Register production models to Unity Catalog for governance

### Tracing
- Enable autolog for supported frameworks
- Add custom spans for business-critical operations
- Use meaningful span names for debugging

### Feature Store
- Use timestamp keys for point-in-time correctness
- Document features with descriptions
- Use Unity Catalog tags for discoverability

### Model Serving
- Enable scale-to-zero for cost optimization
- Use A/B testing for gradual rollouts
- Monitor endpoint metrics and set alerts

### Security
- Never log secrets or API keys
- Use Unity Catalog for access control
- Use service principals for production deployments
