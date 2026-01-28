# Streaming Lakehouse

A production-ready streaming data pipeline using Auto Loader and Structured Streaming for real-time data ingestion into the lakehouse.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Source    │────▶│ Auto Loader │────▶│  Streaming  │────▶│   Delta     │
│  (Files/    │     │  (Ingest)   │     │ Transforms  │     │   Tables    │
│   Kafka)    │     └─────────────┘     └─────────────┘     └─────────────┘
└─────────────┘                                                    │
                                                                   ▼
                                                          ┌─────────────┐
                                                          │  Real-time  │
                                                          │  Analytics  │
                                                          └─────────────┘
```

## What's Included

- **Auto Loader Pipeline** — Incremental file ingestion with schema evolution
- **Streaming Transformations** — Real-time data processing with watermarks
- **Delta Lake Tables** — ACID-compliant storage with time travel
- **Checkpointing** — Fault-tolerant exactly-once processing
- **Monitoring Job** — Stream health monitoring and alerting

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Permissions to create pipelines and jobs
- Source data location (cloud storage or Kafka)

## Quick Start

```bash
# 1. Deploy to your workspace
databricks bundle deploy

# 2. Start the streaming job
databricks bundle run streaming_ingest

# 3. Monitor in the Spark UI or via CLI
databricks jobs list --output json | jq '.[] | select(.settings.name == "streaming-ingest")'
```

## Configuration

Key variables in `databricks.yml`:

| Variable | Description | Default |
|----------|-------------|---------|
| `catalog` | Unity Catalog name | `main` |
| `schema` | Schema for tables | `streaming` |
| `source_path` | Path to streaming source | `/Volumes/.../incoming` |
| `checkpoint_path` | Checkpoint location | `/Volumes/.../checkpoints` |

## Resources Created

- **Job**: `streaming_ingest` — Continuous streaming job
- **Tables**: `raw_events`, `processed_events`, `event_metrics`

## Stream Processing Modes

This exemplar supports two modes:

### Continuous (Default)
```yaml
# In resources/job.yml
continuous: true
```
Stream runs continuously, processing data as it arrives.

### Triggered
```yaml
trigger:
  processing_time: "5 minutes"
```
Micro-batch processing at fixed intervals.

## Testing

```bash
# Run unit tests
uv run pytest tests/

# Integration test with sample data
databricks bundle run streaming_ingest_test
```

## Learn More

- [Auto Loader](https://docs.databricks.com/en/ingestion/auto-loader/index.html)
- [Structured Streaming](https://docs.databricks.com/en/structured-streaming/index.html)
- [Delta Lake Streaming](https://docs.databricks.com/en/delta/delta-streaming.html)
