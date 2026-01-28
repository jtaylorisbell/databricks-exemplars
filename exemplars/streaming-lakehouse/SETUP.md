# Setup: Using This Exemplar as a Template

Follow these steps to use this exemplar as a starting point for your streaming project.

## 1. Copy the Exemplar

```bash
# Copy to your new project location
cp -r exemplars/streaming-lakehouse ~/projects/my-streaming-project
cd ~/projects/my-streaming-project

# Initialize git (optional but recommended)
git init
```

Or use the bootstrap script:

```bash
./shared/scripts/init-from-exemplar.sh streaming-lakehouse my-streaming-project
```

## 2. Required Customizations

### Update `databricks.yml`

Open `databricks.yml` and update:

```yaml
bundle:
  name: my-streaming-project  # Change from streaming-lakehouse

variables:
  catalog:
    default: your_catalog  # Your Unity Catalog name
  schema:
    default: your_schema   # Your schema name
  source_path:
    default: /Volumes/your_catalog/your_schema/incoming  # Your source location
```

### Configure Source Location

Ensure your source data location exists and has appropriate permissions:

```sql
-- Create the volume if needed
CREATE VOLUME IF NOT EXISTS your_catalog.your_schema.incoming;
CREATE VOLUME IF NOT EXISTS your_catalog.your_schema.checkpoints;
```

## 3. Optional Customizations

### Change Processing Mode

In `resources/streaming_job.yml`:

```yaml
# For micro-batch instead of continuous
spark_conf:
  spark.databricks.streaming.trigger.interval: "1 minute"
```

### Adjust Watermark

In `src/stream_processor.py`, modify the watermark for late data handling:

```python
.withWatermark("event_time", "10 minutes")  # Adjust based on your SLA
```

### Add Kafka Source

Replace Auto Loader with Kafka in `src/stream_ingest.py`:

```python
spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_servers) \
    .option("subscribe", "your-topic") \
    .load()
```

## 4. Deploy

```bash
# Validate your configuration
databricks bundle validate

# Deploy to dev (default target)
databricks bundle deploy

# Start the streaming job
databricks bundle run streaming_ingest
```

## 5. Verify

```bash
# Check job status
databricks jobs get --job-id <job-id>

# Query the output table
databricks sql execute --statement "SELECT COUNT(*) FROM your_catalog.your_schema.processed_events"
```

## 6. Monitor

- View stream progress in the Spark UI
- Check the `event_metrics` table for throughput stats
- Set up alerts on the job for failures

## Troubleshooting

### Stream not starting

Check that:
1. Source path exists and has data
2. Checkpoint path is writable
3. You have compute permissions

### Data not appearing

Verify:
1. Source files are in the expected format
2. Schema inference succeeded (check `_schema` location)
3. Watermark isn't filtering all data

### Checkpoint errors

If you need to reset the stream:
```bash
# Warning: This loses exactly-once guarantees
dbutils.fs.rm("/Volumes/.../checkpoints/stream_name", recurse=True)
```
