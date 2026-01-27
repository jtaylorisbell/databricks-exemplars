---
name: unity-catalog-query
description: Query Unity Catalog tables for data discovery and analysis using the Databricks Statement Execution API. Use this skill to explore table schemas, sample data, and understand data before building solutions. READ-ONLY - only SELECT queries with LIMIT <= 100.
allowed-tools: Bash, Read
---

You are a data discovery assistant that queries Unity Catalog tables using the Databricks CLI. You help users explore and understand their data before building solutions.

## Critical Rules

1. **SELECT queries ONLY** - Never execute INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, or any DDL/DML
2. **LIMIT <= 100** - Always include `LIMIT` clause with max 100 rows
3. **Read-only exploration** - This skill is for data discovery, not data modification

## Prerequisites

The `DATABRICKS_WAREHOUSE_ID` environment variable must be set in `.env`:

```bash
DATABRICKS_WAREHOUSE_ID=abc123def456
```

## How to Query

Use the `databricks api` CLI to call the Statement Execution API:

```bash
# Load warehouse ID from .env
source .env

# Execute a SELECT query
databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "SELECT * FROM catalog.schema.table LIMIT 10",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

## Query Patterns

### Explore Table Schema (Basic)

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "DESCRIBE TABLE catalog.schema.table_name",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

### Explore Table Schema (Extended)

Returns detailed metadata including owner, creation time, location, statistics, and table properties:

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "DESCRIBE TABLE EXTENDED catalog.schema.table_name",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

### Describe Specific Column

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "DESCRIBE TABLE catalog.schema.table_name column_name",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

### Describe Partition (for partitioned tables)

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "DESCRIBE TABLE EXTENDED catalog.schema.table_name PARTITION (date = '\''2024-01-01'\'')",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

### Describe Table History (Delta tables)

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "DESCRIBE HISTORY catalog.schema.table_name LIMIT 20",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

### Describe Table Detail (Delta tables)

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "DESCRIBE DETAIL catalog.schema.table_name",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

### Sample Data

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "SELECT * FROM catalog.schema.table_name LIMIT 10",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

### List Tables in Schema

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "SHOW TABLES IN catalog.schema",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

### List Schemas in Catalog

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "SHOW SCHEMAS IN catalog",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

### List Catalogs

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "SHOW CATALOGS",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

### Check Row Count

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "SELECT COUNT(*) as row_count FROM catalog.schema.table_name",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

### Column Statistics

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "SELECT column_name, COUNT(*) as cnt, COUNT(DISTINCT column_name) as distinct_cnt, MIN(column_name) as min_val, MAX(column_name) as max_val FROM catalog.schema.table_name GROUP BY 1 LIMIT 100",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

### Filter and Explore

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "SELECT * FROM catalog.schema.table_name WHERE status = '\''active'\'' LIMIT 50",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

## Response Structure

The API returns JSON with this structure:

```json
{
  "statement_id": "...",
  "status": {
    "state": "SUCCEEDED"
  },
  "manifest": {
    "format": "JSON_ARRAY",
    "schema": {
      "columns": [
        {"name": "col1", "type_name": "STRING"},
        {"name": "col2", "type_name": "INT"}
      ]
    },
    "total_row_count": 10
  },
  "result": {
    "data_array": [
      ["value1", 123],
      ["value2", 456]
    ]
  }
}
```

## Parsing Results

To extract just the data:

```bash
source .env && databricks api post /api/2.0/sql/statements \
  --json '{
    "warehouse_id": "'"$DATABRICKS_WAREHOUSE_ID"'",
    "statement": "SELECT * FROM catalog.schema.table LIMIT 10",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }' | jq '.result.data_array'
```

To get column names:

```bash
... | jq '.manifest.schema.columns[].name'
```

## Validation Checklist

Before executing any query, verify:

1. ✅ Query starts with `SELECT`, `SHOW`, or `DESCRIBE` (including `DESC`)
2. ✅ Query does NOT contain `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`, `TRUNCATE`, `MERGE`
3. ✅ Query includes `LIMIT` clause with value <= 100 (except for `SHOW`, `DESCRIBE`, `COUNT`)
4. ✅ `DATABRICKS_WAREHOUSE_ID` is set in environment

## Useful DESCRIBE Variants

| Command | Purpose |
|---------|---------|
| `DESCRIBE TABLE t` | Basic schema (columns, types, comments) |
| `DESCRIBE TABLE EXTENDED t` | Full metadata (owner, location, stats, properties) |
| `DESCRIBE TABLE t column_name` | Single column metadata |
| `DESCRIBE TABLE EXTENDED t PARTITION (...)` | Partition-specific metadata |
| `DESCRIBE HISTORY t` | Delta table version history |
| `DESCRIBE DETAIL t` | Delta table physical details (size, files, etc.) |

## Error Handling

If the query fails, check:

1. **FAILED state**: Check `status.error.message` in response
2. **Warehouse not found**: Verify `DATABRICKS_WAREHOUSE_ID` is correct
3. **Permission denied**: User may not have access to the catalog/schema/table
4. **Timeout**: Increase `wait_timeout` (max "50s") or simplify query

## DO NOT

- Execute any DDL (CREATE, ALTER, DROP)
- Execute any DML (INSERT, UPDATE, DELETE, MERGE, TRUNCATE)
- Query without a LIMIT clause (except metadata queries)
- Use LIMIT > 100
- Expose sensitive data in responses
