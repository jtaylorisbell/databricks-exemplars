# Getting Started

This guide covers the prerequisites and setup needed before using any exemplar.

## Prerequisites

### 1. Databricks CLI

Install the Databricks CLI:

```bash
# macOS/Linux
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

# Or with Homebrew
brew tap databricks/tap
brew install databricks
```

Verify installation:

```bash
databricks --version
```

### 2. Databricks Workspace Access

You need access to a Databricks workspace with:

- Permission to create jobs, pipelines, or other resources (depending on the exemplar)
- Unity Catalog access (most exemplars use Unity Catalog)
- Appropriate compute permissions

### 3. Authentication (OAuth)

All exemplars use OAuth for authentication. **Never use Personal Access Tokens (PATs).**

#### Configure OAuth with the CLI

```bash
databricks auth login --host https://your-workspace.cloud.databricks.com
```

This opens a browser for OAuth authentication and stores credentials in `~/.databrickscfg`.

#### Verify Authentication

```bash
databricks current-user me
```

You should see your user information returned.

### 4. Python Environment (for Python-based exemplars)

We recommend using `uv` for Python environment management:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv

# Activate it
source .venv/bin/activate
```

## Workspace Configuration

### Unity Catalog Setup

Most exemplars expect Unity Catalog to be enabled. Ensure you have:

1. A catalog you can use (or permission to create one)
2. A schema within that catalog (or permission to create one)
3. Appropriate grants for your user/service principal

### Compute Configuration

Exemplars use different compute types:

| Compute Type | Used By | Notes |
|--------------|---------|-------|
| Serverless | Most exemplars | No cluster management needed |
| Jobs Compute | Batch pipelines | Auto-created by jobs |
| SQL Warehouse | Analytics exemplars | Serverless recommended |

## Common Bundle Commands

Once you've copied an exemplar and configured it:

```bash
# Validate the bundle configuration
databricks bundle validate

# Deploy to your workspace (dev target)
databricks bundle deploy

# Deploy to a specific target
databricks bundle deploy --target prod

# Run a job defined in the bundle
databricks bundle run <job-name>

# Destroy deployed resources
databricks bundle destroy
```

## Troubleshooting

### "Not authenticated" errors

Re-run OAuth login:

```bash
databricks auth login --host https://your-workspace.cloud.databricks.com
```

### "Permission denied" errors

Check that your user has the required permissions in Unity Catalog and on the workspace.

### Bundle validation errors

Run with verbose output:

```bash
databricks bundle validate --debug
```

## Next Steps

1. Browse the [exemplars](../exemplars/) to find one matching your use case
2. Follow that exemplar's `SETUP.md` to use it as a template
3. Customize and deploy to your workspace
