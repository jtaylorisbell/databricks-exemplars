# Batch ETL Pipeline Exemplar

A production-ready batch ETL pipeline demonstrating the **medallion architecture** (bronze → silver → gold) using Databricks Lakeflow Declarative Pipelines.

## What This Exemplar Does

This pipeline transforms bakery sales data from `samples.bakehouse` into analytics-ready tables:

```
SOURCE                     BRONZE                  SILVER                 GOLD
samples.bakehouse          (raw + audit)           (cleaned)              (aggregates)
─────────────────          ─────────────           ─────────────          ─────────────────────
sales_transactions    ──▶  bronze_transactions ──▶ silver_transactions ──▶ gold_daily_sales_summary
sales_customers       ──▶  bronze_customers    ──▶ silver_customers    ──▶ gold_customer_360
sales_franchises      ──▶  bronze_franchises   ──▶ silver_franchises   ──▶ gold_product_performance
sales_suppliers       ──▶  bronze_suppliers                               gold_franchise_performance
                                                                          gold_payment_analysis
```

### Source Data (samples.bakehouse)

| Table | Rows | Description |
|-------|------|-------------|
| `sales_transactions` | 3,333 | Sales from April-May 2024 |
| `sales_customers` | 300 | Customer demographics |
| `sales_franchises` | 48 | Bakery franchise locations |
| `sales_suppliers` | 27 | Ingredient suppliers |

### Output Tables

#### Bronze Layer (Raw + Audit)
- `bronze_transactions` — Raw transactions with `_ingested_at` timestamp
- `bronze_customers` — Raw customer data with audit metadata
- `bronze_franchises` — Raw franchise locations
- `bronze_suppliers` — Raw supplier information

#### Silver Layer (Cleaned + Enriched)
- `silver_transactions` — Validated transactions with:
  - PII masking (card numbers)
  - Date component extraction
  - Standardized payment methods
  - Data quality enforcement
- `silver_customers` — Standardized with:
  - Full name derivation
  - Regional segmentation (NA/EU/APAC/OTHER)
- `silver_franchises` — Enriched with supplier relationships

#### Gold Layer (Business Aggregates)
- `gold_daily_sales_summary` — Daily revenue, transactions, customers by franchise
- `gold_customer_360` — Customer lifetime value and behavior metrics
- `gold_product_performance` — Product-level sales analysis
- `gold_franchise_performance` — Franchise KPIs with location data
- `gold_payment_analysis` — Payment method trends

---

## What This Exemplar Does NOT Do

| Not Included | Why | What to Use Instead |
|--------------|-----|---------------------|
| **Streaming ingestion** | This is a batch exemplar | See `streaming-lakehouse` exemplar |
| **Raw file ingestion** | Source is already Delta tables | Use Auto Loader for raw files |
| **CDC/SCD Type 2** | Sample data is static | Use `create_auto_cdc_flow()` for CDC |
| **Data masking policies** | Requires Unity Catalog setup | Use column masks + row filters |
| **ML feature engineering** | Out of scope | See `ml-model-serving` exemplar |
| **Custom scheduling** | Disabled by default | Uncomment `schedule` in job.yml |
| **Alerting/notifications** | Environment-specific | Configure `email_notifications` |

---

## Design Choices

### 1. Materialized Views vs Streaming Tables

**Choice**: All tables are `@dp.materialized_view()`, not `@dp.table()` (streaming).

**Rationale**: The source data (`samples.bakehouse`) is static Delta tables, not a streaming source. Materialized views are appropriate for:
- Batch processing of existing tables
- Full recomputation on each refresh
- Simpler debugging and development

**When to change**: Use `@dp.table()` with `spark.readStream` when:
- Ingesting from Auto Loader (cloud files)
- Processing Kafka/Event Hubs streams
- Implementing append-only patterns

### 2. Single Pipeline File vs Multiple Files

**Choice**: All layers defined in a single `pipeline.py` file.

**Rationale**:
- Easier to understand data lineage at a glance
- Constants and rules shared across layers
- Simpler deployment (one file to manage)

**When to change**: Split into separate files when:
- Multiple teams own different layers
- Pipeline exceeds ~500 lines
- Different testing strategies per layer

### 3. Data Quality: Drop vs Fail

**Choice**: Invalid records are dropped (`@dp.expect_all_or_drop`), not failed.

**Rationale**:
- Pipeline continues processing valid data
- Bad records don't block downstream analytics
- Quality metrics visible in pipeline UI

**When to change**: Use `@dp.expect_or_fail()` when:
- Data quality is critical (financial, compliance)
- Bad data indicates upstream system failure
- You need to halt processing for investigation

### 4. PII Handling

**Choice**: Card numbers are masked in silver layer (last 4 digits only).

**Rationale**:
- Bronze preserves raw data for debugging
- Silver enforces PII protection
- Gold never sees raw PII

**When to change**: For production PII handling:
- Use Unity Catalog column masks
- Implement row-level security
- Consider separate PII and analytics schemas

### 5. No Partitioning

**Choice**: Tables are not partitioned.

**Rationale**:
- Sample data is small (~3K rows)
- Liquid clustering is preferred for modern Delta
- Avoids small file problems

**When to change**: Add partitioning when:
- Data exceeds millions of rows
- Queries filter on predictable columns (date)
- Using `partition_cols=["transaction_date"]`

### 6. Serverless Compute

**Choice**: Pipeline runs on serverless compute by default (`serverless: true` in `resources/pipeline.yml`).

**Rationale**:
- No cluster management overhead
- Automatic scaling and optimization
- Cost-efficient for batch workloads
- Faster startup times

**When to change**: Use classic compute clusters when:
- You need specific Spark configurations not available in serverless
- Your workload requires GPUs
- You have compliance requirements for dedicated compute

To switch to classic compute, edit `resources/pipeline.yml`:

```yaml
resources:
  pipelines:
    batch_etl_pipeline:
      # ... other settings ...
      serverless: false  # Disable serverless
      clusters:
        - label: default
          autoscale:
            min_workers: 1
            max_workers: 4
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Pipeline Framework | Lakeflow Declarative Pipelines (formerly DLT) |
| Python API | `pyspark.pipelines` (Spark 4.1+) |
| Compute | Photon-enabled serverless |
| Storage | Delta Lake on Unity Catalog |
| Orchestration | Databricks Asset Bundles |
| Scheduling | Databricks Workflows (optional) |

---

## Quick Start

### Prerequisites

**1. Configure Databricks CLI authentication**

The bundle needs to know which workspace to deploy to. Use one of these methods:

```bash
# Option A: CLI profile (recommended)
databricks auth login --host https://your-workspace.cloud.databricks.com --profile my-profile

# Option B: Environment variable
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
```

**2. Specify catalog and schema**

This bundle requires a Unity Catalog `catalog` and `schema` for output tables (no defaults).

### Deploy and Run

```bash
# 1. Navigate to exemplar
cd exemplars/batch-etl-pipeline

# 2. Validate bundle (specify your catalog and schema)
databricks bundle validate --profile YOUR_PROFILE \
  --var="catalog=your_catalog,schema=your_schema"

# 3. Deploy to workspace
databricks bundle deploy --profile YOUR_PROFILE \
  --var="catalog=your_catalog,schema=your_schema"

# 4. Run the pipeline
databricks bundle run batch_etl_pipeline --profile YOUR_PROFILE \
  --var="catalog=your_catalog,schema=your_schema"
```

### Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `catalog` | Yes | Unity Catalog name for output tables |
| `schema` | Yes | Schema name for pipeline tables |

You can set variables in three ways:

1. **Command line** (recommended for testing):
   ```bash
   --var="catalog=my_catalog,schema=my_schema"
   ```

2. **Environment variables**:
   ```bash
   export BUNDLE_VAR_catalog=my_catalog
   export BUNDLE_VAR_schema=my_schema
   databricks bundle deploy --profile YOUR_PROFILE
   ```

3. **In databricks.yml** (for fixed defaults):
   ```yaml
   variables:
     catalog:
       default: my_catalog
     schema:
       default: my_schema
   ```

See [SETUP.md](./SETUP.md) for detailed setup instructions.

---

## File Structure

```
batch-etl-pipeline/
├── databricks.yml           # Bundle configuration
├── pyproject.toml           # Python dependencies (for local testing)
├── README.md                # This file
├── SETUP.md                 # Detailed setup guide
├── src/
│   └── pipeline.py          # Pipeline definition (bronze/silver/gold)
├── resources/
│   ├── pipeline.yml         # Lakeflow pipeline resource
│   └── job.yml              # Workflow job resource
└── tests/
    └── test_transformations.py  # Unit tests for transformation logic
```

---

## Running Tests Locally

```bash
# Install dev dependencies
uv sync --extra dev

# Run unit tests
uv run pytest tests/ -v
```

Note: Pipeline code runs only in Databricks. Local tests validate transformation logic using local Spark.

---

## Customization Guide

### Set Target Catalog/Schema

Variables are required and can be set in multiple ways:

**Option 1: Command line (per-command)**
```bash
databricks bundle deploy --var="catalog=my_catalog,schema=my_schema"
```

**Option 2: Environment variables (session-wide)**
```bash
export BUNDLE_VAR_catalog=my_catalog
export BUNDLE_VAR_schema=my_schema
databricks bundle deploy
```

**Option 3: Add defaults to databricks.yml (permanent)**
```yaml
variables:
  catalog:
    description: Unity Catalog name for output tables (required)
    default: my_catalog  # Add this line
  schema:
    description: Schema name for pipeline tables (required)
    default: my_schema   # Add this line
```

### Add New Gold Table

1. Add function to `src/pipeline.py`:

```python
@dp.materialized_view(
    name="gold_my_new_aggregate",
    comment="Description of what this table provides",
)
def gold_my_new_aggregate():
    return (
        spark.read.table("silver_transactions")
        .groupBy("some_column")
        .agg(F.count("*").alias("count"))
    )
```

2. Redeploy: `databricks bundle deploy --var="catalog=...,schema=..."`

### Enable Scheduling

Uncomment in `resources/job.yml`:

```yaml
schedule:
  quartz_cron_expression: "0 0 6 * * ?"  # 6 AM UTC daily
  timezone_id: UTC
```

### Switch to Classic Compute

By default, this exemplar uses **serverless compute**. To use classic clusters instead, edit `resources/pipeline.yml`:

```yaml
resources:
  pipelines:
    batch_etl_pipeline:
      name: batch-etl-pipeline-${bundle.target}
      catalog: ${var.catalog}
      target: ${var.schema}
      development: true
      continuous: false
      channel: PREVIEW
      photon: true
      serverless: false  # ← Disable serverless
      libraries:
        - file:
            path: ../src/pipeline.py
      clusters:            # ← Add cluster configuration
        - label: default
          autoscale:
            min_workers: 1
            max_workers: 4
          # Optional: specify node types
          # node_type_id: Standard_DS3_v2
          # driver_node_type_id: Standard_DS3_v2
```

---

## References

- [Lakeflow Declarative Pipelines Documentation](https://docs.databricks.com/aws/en/ldp/index.html)
- [pyspark.pipelines Python API](https://docs.databricks.com/aws/en/ldp/developer/python-ref)
- [Databricks Asset Bundles](https://docs.databricks.com/aws/en/dev-tools/bundles/index.html)
- [Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture)
