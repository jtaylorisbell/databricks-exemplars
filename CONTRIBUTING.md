# Contributing to Databricks Exemplars

Thank you for contributing! This guide explains how to add new exemplars to the repository.

## Exemplar Requirements

Every exemplar must be:

1. **Self-contained** — Users can copy the folder and have everything needed
2. **Production-ready** — Code follows best practices, includes error handling, and is tested
3. **Well-documented** — Clear README, setup instructions, and inline comments
4. **Deployable** — Includes a working Databricks Asset Bundle configuration

## Required Files

Each exemplar must include:

```
exemplars/your-exemplar-name/
├── README.md            # Required: Overview, architecture, prerequisites
├── SETUP.md             # Required: Step-by-step template usage guide
├── databricks.yml       # Required: DAB bundle definition
├── src/                 # Required: Source code
├── resources/           # Required: DAB resource definitions (jobs, pipelines, etc.)
└── tests/               # Required: Unit and/or integration tests
```

## README.md Template

Your exemplar's README should include:

1. **Title and one-line description**
2. **Architecture diagram** (can be ASCII or image)
3. **What's included** — List of components/resources
4. **Prerequisites** — Specific requirements beyond the common ones
5. **Quick start** — Shortest path to seeing it work
6. **Configuration options** — What can be customized
7. **Testing** — How to run tests

## SETUP.md Template

The SETUP.md should guide users through using the exemplar as a starting point:

1. **Copy the exemplar** — Exact commands to copy
2. **Required customizations** — What must be changed (bundle name, catalog, etc.)
3. **Optional customizations** — What can be changed
4. **Deploy** — How to deploy to their workspace
5. **Verify** — How to confirm it's working
6. **Next steps** — Common modifications users might want

## databricks.yml Guidelines

- Use variables for environment-specific values
- Include sensible defaults where possible
- Support multiple deployment targets (dev, staging, prod)
- Use OAuth for authentication (never PATs)

Example structure:

```yaml
bundle:
  name: your-exemplar-name

variables:
  catalog:
    description: Unity Catalog name
    default: main
  schema:
    description: Schema name
    default: default

targets:
  dev:
    mode: development
    default: true
    workspace:
      host: ${var.workspace_host}

  prod:
    mode: production
    workspace:
      host: ${var.workspace_host}
```

## Code Quality Standards

### Python

- Type hints on all function signatures
- Docstrings for all public functions
- PEP 8 compliant
- Meaningful variable names reflecting business concepts

### SQL

- Uppercase keywords
- Clear CTEs over nested subqueries
- Comments explaining complex logic

### Testing

- Unit tests for business logic
- Integration tests for pipeline components
- Test data that doesn't require real credentials

## Submission Process

1. **Create a branch** from `main`
2. **Add your exemplar** following the structure above
3. **Test deployment** to ensure it works end-to-end
4. **Submit a PR** with:
   - Description of what the exemplar demonstrates
   - Any prerequisites reviewers need to test it
   - Screenshots or output showing it working

## Questions?

Open an issue if you have questions about contributing.
