# Exemplars

Each folder contains a complete, self-contained Databricks solution that you can use as a starting point for your own project.

## Available Exemplars

| Exemplar | Description | Complexity |
|----------|-------------|------------|
| [batch-etl-pipeline](./batch-etl-pipeline/) | Medallion architecture ETL with Lakeflow Declarative Pipelines | Beginner |
| [streaming-lakehouse](./streaming-lakehouse/) | Real-time ingestion with Auto Loader and Structured Streaming | Intermediate |
| [ml-model-serving](./ml-model-serving/) | End-to-end ML workflow with MLflow and Model Serving | Advanced |

## How to Use

1. **Pick an exemplar** that matches your use case
2. **Read its README.md** to understand what it does
3. **Follow its SETUP.md** to use it as a template for your project

## Exemplar Structure

Every exemplar follows the same structure:

```
exemplar-name/
├── README.md           # What it does, architecture, prerequisites
├── SETUP.md            # How to use this as a template
├── databricks.yml      # Databricks Asset Bundle definition
├── src/                # Source code (notebooks, Python modules, SQL)
├── resources/          # DAB resource definitions (jobs, pipelines)
└── tests/              # Unit and integration tests
```

## Comparison Guide

### Choose `batch-etl-pipeline` if you need:
- Scheduled data processing (daily, hourly)
- Medallion architecture (bronze → silver → gold)
- Simple orchestration with Lakeflow

### Choose `streaming-lakehouse` if you need:
- Real-time or near-real-time data processing
- Continuous ingestion from files or message queues
- Low-latency data availability

### Choose `ml-model-serving` if you need:
- Training and deploying ML models
- Feature engineering and management
- Real-time model inference endpoints
