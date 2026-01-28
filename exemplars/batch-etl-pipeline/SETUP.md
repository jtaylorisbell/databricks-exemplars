# Setup Guide

This guide shows how to deploy this exemplar to your Databricks workspace, and how to use it as a starting point for your own project.

## Prerequisites

Before you begin:

1. **Databricks CLI** installed ([docs](https://docs.databricks.com/aws/en/dev-tools/cli/install.html))
2. **Workspace authentication** configured (see below)
3. **Unity Catalog** enabled on your workspace
4. **Permissions** to create pipelines, jobs, and tables

### Configure Workspace Authentication

The bundle needs to know which workspace to deploy to. Choose one method:

**Option A: CLI Profile (recommended for multiple workspaces)**

```bash
# Create a named profile with OAuth authentication
databricks auth login --host https://your-workspace.cloud.databricks.com --profile my-profile

# Verify it works
databricks current-user me --profile my-profile

# Use with bundle commands
databricks bundle deploy --profile my-profile
```

**Option B: Environment Variable (simpler for single workspace)**

```bash
# Set the workspace host
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com

# Authenticate
databricks auth login

# Verify
databricks current-user me

# Bundle commands will use DATABRICKS_HOST automatically
databricks bundle deploy
```

**Option C: Default Profile**

If you have a `[DEFAULT]` profile in `~/.databrickscfg`, bundle commands will use it automatically when no `--profile` is specified.

---

## Option A: Deploy the Exemplar Directly

Use this option to explore the exemplar without modification.

### Step 1: Choose Your Catalog and Schema

The exemplar writes tables to Unity Catalog. You need a catalog and schema with write permissions.

Variables are **required** and must be provided via `--var` flag or environment variables.

### Step 2: Validate

```bash
cd exemplars/batch-etl-pipeline

databricks bundle validate --profile YOUR_PROFILE \
  --var="catalog=your_catalog,schema=your_schema"
```

You should see:

```
Name: batch-etl-pipeline
Target: dev
Workspace:
  Host: https://your-workspace.cloud.databricks.com
  ...

Validation OK!
```

### Step 3: Deploy

```bash
databricks bundle deploy --profile YOUR_PROFILE \
  --var="catalog=your_catalog,schema=your_schema"
```

This creates:
- A Lakeflow pipeline named `batch-etl-pipeline-dev`
- A workflow job named `batch-etl-refresh-dev`

### Step 4: Run

```bash
databricks bundle run batch_etl_pipeline --profile YOUR_PROFILE \
  --var="catalog=your_catalog,schema=your_schema"
```

Or trigger via the Databricks UI:
1. Navigate to **Workflows** → **Delta Live Tables**
2. Find `batch-etl-pipeline-dev`
3. Click **Start**

**Tip:** To avoid repeating `--var` on every command, set environment variables:

```bash
export BUNDLE_VAR_catalog=your_catalog
export BUNDLE_VAR_schema=your_schema

# Now you can run without --var
databricks bundle deploy --profile YOUR_PROFILE
databricks bundle run batch_etl_pipeline --profile YOUR_PROFILE
```

### Step 5: Verify

Check that tables were created:

```sql
-- In a Databricks notebook or SQL editor
USE CATALOG your_catalog;
USE SCHEMA your_schema;

SHOW TABLES;
-- Should show: bronze_*, silver_*, gold_*

SELECT * FROM gold_daily_sales_summary LIMIT 10;
```

---

## Option B: Use as a Template for Your Project

Use this option to create a new project based on this exemplar.

### Step 1: Copy the Exemplar

```bash
# Using the bootstrap script (recommended)
./shared/scripts/init-from-exemplar.sh batch-etl-pipeline my-sales-pipeline

# Or manually
cp -r exemplars/batch-etl-pipeline ~/projects/my-sales-pipeline
cd ~/projects/my-sales-pipeline
git init
```

### Step 2: Rename the Bundle

Edit `databricks.yml`:

```yaml
bundle:
  name: my-sales-pipeline  # ← Your project name

variables:
  catalog:
    default: your_catalog
  schema:
    default: your_schema
```

### Step 3: Update Resource Names

Edit `resources/pipeline.yml`:

```yaml
resources:
  pipelines:
    my_pipeline:  # ← Rename from batch_etl_pipeline
      name: my-sales-pipeline-${bundle.target}
      # ...
```

Edit `resources/job.yml`:

```yaml
resources:
  jobs:
    my_refresh_job:  # ← Rename from batch_etl_refresh
      name: my-sales-pipeline-refresh-${bundle.target}
      tasks:
        - task_key: refresh_pipeline
          pipeline_task:
            pipeline_id: ${resources.pipelines.my_pipeline.id}  # ← Match pipeline name
```

### Step 4: Modify the Pipeline Code

Edit `src/pipeline.py` to read from your data sources:

```python
# Change the source configuration
SOURCE_CATALOG = "your_source_catalog"
SOURCE_SCHEMA = "your_source_schema"

# Or read from files
@dp.materialized_view(name="bronze_events")
def bronze_events():
    return spark.read.format("json").load("/path/to/your/data")
```

### Step 5: Update Data Quality Rules

Edit the rules in `src/pipeline.py` to match your data:

```python
TRANSACTION_RULES = {
    "valid_id": "id IS NOT NULL",
    "valid_amount": "amount > 0",
    # Add your rules...
}
```

### Step 6: Deploy Your Project

```bash
cd ~/projects/my-sales-pipeline
databricks bundle deploy --profile YOUR_PROFILE
```

---

## Configuration Reference

### Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `catalog` | **Yes** | Unity Catalog name for output tables |
| `schema` | **Yes** | Schema name for output tables |

Set variables using one of these methods:

```bash
# Method 1: Command line flag
--var="catalog=my_catalog,schema=my_schema"

# Method 2: Environment variables
export BUNDLE_VAR_catalog=my_catalog
export BUNDLE_VAR_schema=my_schema
```

### Targets

| Target | Mode | Description |
|--------|------|-------------|
| `dev` | development | Development mode (default) |
| `prod` | production | Production mode |

Deploy to production:

```bash
databricks bundle deploy --target prod --profile YOUR_PROFILE \
  --var="catalog=prod_catalog,schema=prod_schema"
```

### Pipeline Settings

In `resources/pipeline.yml`:

| Setting | Description | Default |
|---------|-------------|---------|
| `development` | Development mode (relaxed validation) | `true` |
| `continuous` | Run continuously vs triggered | `false` |
| `photon` | Use Photon acceleration | `true` |
| `channel` | Release channel | `PREVIEW` |

---

## Common Customizations

### Add Scheduling

Uncomment in `resources/job.yml`:

```yaml
schedule:
  quartz_cron_expression: "0 0 6 * * ?"  # Daily at 6 AM UTC
  timezone_id: UTC
```

### Add Email Notifications

```yaml
email_notifications:
  on_failure:
    - team@example.com
  on_success:
    - team@example.com
```

### Enable Liquid Clustering

In `src/pipeline.py`, add to your table decorators:

```python
@dp.materialized_view(
    name="silver_transactions",
    cluster_by=["transaction_date", "franchiseID"],  # ← Add this
)
```

### Add Partitioning

```python
@dp.materialized_view(
    name="gold_daily_sales_summary",
    partition_cols=["transaction_date"],  # ← Add this
)
```

---

## Troubleshooting

### "Catalog not found" or "Schema not found"

Create the schema if it doesn't exist:

```sql
CREATE CATALOG IF NOT EXISTS your_catalog;
CREATE SCHEMA IF NOT EXISTS your_catalog.batch_etl;
```

Or grant yourself access:

```sql
GRANT ALL PRIVILEGES ON CATALOG your_catalog TO `your-email@example.com`;
```

### "Pipeline validation failed"

Run with debug output:

```bash
databricks bundle validate --debug
```

Common causes:
- Invalid Python syntax in pipeline.py
- Missing source tables
- Permission issues

### "samples.bakehouse not found"

The `samples` catalog is available on most Databricks workspaces. If not:

1. Check if it exists: `SHOW CATALOGS`
2. If missing, contact your workspace admin
3. Or modify the pipeline to use a different data source

### "No compute resources available"

Ensure you have permission to use serverless compute or create clusters. Check with your workspace admin.

### Pipeline runs but no data

1. Check the pipeline UI for errors in individual tables
2. Look at the data quality metrics (dropped records)
3. Verify source tables have data:

```sql
SELECT COUNT(*) FROM samples.bakehouse.sales_transactions;
```

---

## Cleanup

To remove all deployed resources:

```bash
databricks bundle destroy --profile YOUR_PROFILE
```

This removes the pipeline and job but **does not delete the tables**. To fully clean up:

```sql
DROP SCHEMA IF EXISTS your_catalog.batch_etl CASCADE;
```
