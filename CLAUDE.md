# Project Instructions

## Databricks Documentation Navigation

When researching Databricks topics, the documentation (docs.databricks.com) is organized into these key areas:

| Area | Topics Covered |
|------|----------------|
| **Getting Started** | Quick-start guides for querying data, ETL pipelines, ML models, AI agents, free trial |
| **Core Infrastructure** | Data lakehouse architecture, Delta Lake, Unity Catalog, compute options (serverless, GPU) |
| **Data Engineering** | Lakeflow pipeline orchestration, ingestion connectors, streaming, job scheduling |
| **Analytics & ML** | SQL warehousing, generative AI, vector search, model serving, hyperparameter tuning |
| **Security & Governance** | Access control, Unity Catalog privileges, row/column filtering, encryption, compliance (HIPAA, FedRAMP, PCI-DSS) |
| **Developer Tools** | SDKs (Python, Java, Go), REST APIs, CLI, Git integration, Terraform |
| **Integrations** | BI tools (Tableau, Power BI), data platforms (Fivetran, Kafka, Airflow), dbt |

The documentation also includes migration guides, best practices, troubleshooting resources, and complete SQL/API references.

## Authentication
- **ALWAYS use OAuth for Databricks authentication** - Never use Personal Access Tokens (PATs)
- Use service principals with OAuth for automated workloads

## Code Style

### Python
- Use type hints in all function signatures
- Include docstrings for all functions
- Follow PEP 8 style guidelines
- Use meaningful variable names that reflect business concepts
- Add comments explaining complex business logic
