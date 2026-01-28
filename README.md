# Databricks Exemplars

Gold-standard, production-ready examples of Databricks solutions. Each exemplar is fully independent and includes [Databricks Asset Bundle](https://docs.databricks.com/dev-tools/bundles/index.html) definitions for easy deployment.

## Quick Start

1. **Browse the exemplars** in the [`exemplars/`](./exemplars/) directory
2. **Pick one** that matches your use case
3. **Follow its `SETUP.md`** to use it as a starting point for your project

Or use the bootstrap script:

```bash
./shared/scripts/init-from-exemplar.sh <exemplar-name> <your-project-name>
```

## Exemplar Catalog

| Exemplar | Use Case | Key Services | Complexity |
|----------|----------|--------------|------------|
| [batch-etl-pipeline](./exemplars/batch-etl-pipeline/) | Scheduled batch data processing with medallion architecture | Lakeflow Declarative Pipelines, Unity Catalog, Delta Lake | Beginner |
| [streaming-lakehouse](./exemplars/streaming-lakehouse/) | Real-time data ingestion and processing | Structured Streaming, Delta Lake, Auto Loader | Intermediate |
| [ml-model-serving](./exemplars/ml-model-serving/) | End-to-end ML with model serving | MLflow, Feature Store, Model Serving | Advanced |

## Prerequisites

Before using any exemplar, ensure you have:

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) installed and configured
- Access to a Databricks workspace with appropriate permissions
- OAuth configured for authentication (see [Getting Started](./docs/getting-started.md))

## Repository Structure

```
databricks-exemplars/
├── README.md                 # This file
├── CONTRIBUTING.md           # Guidelines for adding new exemplars
├── docs/                     # Common documentation
│   └── getting-started.md    # Setup prerequisites and authentication
├── exemplars/                # All exemplar projects
│   ├── batch-etl-pipeline/
│   ├── streaming-lakehouse/
│   └── ml-model-serving/
└── shared/                   # Reusable utilities
    └── scripts/
```

## Using an Exemplar as a Template

Each exemplar is designed to be copied and customized. The general workflow:

1. **Copy** the exemplar folder to your new project location
2. **Update** `databricks.yml` with your bundle name and workspace settings
3. **Customize** the code in `src/` for your specific use case
4. **Deploy** with `databricks bundle deploy`

See individual exemplar `SETUP.md` files for detailed instructions.

## Contributing

Want to add a new exemplar? See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

[Add your license here]
