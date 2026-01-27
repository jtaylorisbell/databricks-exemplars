---
name: databricks-app-backend
description: Expert in building application backends deployed as Databricks Apps using Python, FastAPI, and Pydantic. Use this skill for developing APIs, connecting to DBSQL warehouses or Lakebase, configuring app.yaml, and following Databricks Apps best practices.
---

You are an expert backend developer specializing in building applications deployed as Databricks Apps. You use Python, FastAPI, and Pydantic exclusively. You understand how to connect to Databricks data sources properly.

## Core Technology Stack

- **Framework**: FastAPI
- **Validation**: Pydantic v2, pydantic-settings
- **Database (Analytical/Heavy Queries)**: Databricks SQL Warehouses via `databricks-sql-connector`
- **Database (OLTP/Transactional)**: Lakebase via `psycopg` (v3, NOT psycopg2)
- **SDK**: `databricks-sdk` (>=0.40.0) for workspace operations and credential generation
- **Logging**: `structlog` for structured logging

## Critical Rules

**NEVER use local databases in development:**
- Do NOT spin up local PostgreSQL, SQLite, or any other local database
- ALWAYS connect to real DBSQL warehouses or Lakebase instances
- Use environment variables to configure connections for different environments

**Choose the right database for the workload:**
| Workload Type | Use This | Why |
|---------------|----------|-----|
| Heavy analytical queries, aggregations, large scans | DBSQL Warehouse | Optimized for OLAP, scales for big data |
| Transactional operations, CRUD, low-latency reads/writes | Lakebase | PostgreSQL-compatible OLTP with autoscaling |

## Project Structure

```
my-app/
├── app.py                   # Entry point for Databricks Apps
├── app.yaml                 # Databricks Apps configuration
├── pyproject.toml           # Project config (use uv)
├── requirements.txt         # Generated via: uv export --no-hashes --no-emit-project
├── src/
│   └── my_app/
│       ├── __init__.py
│       ├── config.py        # Settings and OAuth token management
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py      # FastAPI application
│       │   └── schemas.py   # Pydantic request/response models
│       ├── db/
│       │   ├── __init__.py
│       │   ├── postgres.py  # Lakebase connection factory
│       │   └── schemas.py   # DB models
│       └── core/
│           ├── __init__.py
│           └── service.py   # Business logic
└── tests/
```

## app.yaml Configuration

Keep it simple - resources are configured in the Databricks Apps UI, not in app.yaml.

Note: Use `DATABRICKS_APP_PORT` environment variable (automatically set by the platform) or hardcode port 8000:

```yaml
command:
  - uvicorn
  - app:app
  - --host
  - "0.0.0.0"
  - --port
  - "8000"

env:
  - name: LOG_LEVEL
    value: "INFO"
```

## Entry Point (app.py)

```python
"""Databricks Apps entry point."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and expose the FastAPI app
from my_app.api.main import app  # noqa: E402

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Configuration with OAuth Token Management (config.py)

```python
"""Configuration management with OAuth token refresh."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from functools import lru_cache
from typing import ClassVar

import structlog
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger()


class OAuthTokenManager:
    """Manages OAuth tokens for Lakebase with automatic refresh."""

    _instance: ClassVar["OAuthTokenManager | None"] = None
    _token: str | None = None
    _expires_at: datetime | None = None
    _instance_name: str | None = None

    def __new__(cls) -> "OAuthTokenManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_token(
        self,
        instance_name: str,
        workspace_host: str | None = None,
        force_refresh: bool = False,
    ) -> str | None:
        """Get a valid OAuth token, refreshing if necessary."""
        if not instance_name:
            return None

        # Check if we have a valid cached token (with 5 min buffer)
        if (
            not force_refresh
            and self._token
            and self._instance_name == instance_name
            and self._expires_at
            and datetime.now() < self._expires_at - timedelta(minutes=5)
        ):
            return self._token

        # Generate new token
        try:
            from databricks.sdk import WorkspaceClient

            logger.info("generating_oauth_token", instance=instance_name)
            w = WorkspaceClient(host=workspace_host) if workspace_host else WorkspaceClient()
            cred = w.database.generate_database_credential(
                request_id=str(uuid.uuid4()),
                instance_names=[instance_name],
            )

            self._token = cred.token
            self._instance_name = instance_name
            self._expires_at = datetime.now() + timedelta(minutes=55)

            logger.info("oauth_token_generated", expires_at=self._expires_at.isoformat())
            return self._token

        except Exception as e:
            logger.error("oauth_token_generation_failed", error=str(e))
            return None


_token_manager = OAuthTokenManager()


class DatabricksSettings(BaseSettings):
    """Databricks workspace connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="DATABRICKS_",
        env_file=".env",
        extra="ignore",
    )

    host: str = ""


class LakebaseSettings(BaseSettings):
    """Lakebase database configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LAKEBASE_",
        env_file=".env",
        extra="ignore",
    )

    host: str = ""
    port: int = 5432
    database: str = "databricks_postgres"
    user: str = ""
    instance_name: str = ""  # For OAuth token generation


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"

    @property
    def lakebase(self) -> LakebaseSettings:
        return LakebaseSettings()

    @property
    def databricks(self) -> DatabricksSettings:
        return DatabricksSettings()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
```

## Lakebase Connection Factory (db/postgres.py)

This pattern detects whether running in Databricks Apps or locally:

```python
"""PostgreSQL database client for Lakebase."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import psycopg
import structlog

logger = structlog.get_logger()


class LakebaseConnectionFactory:
    """Factory for creating Lakebase connections with OAuth authentication.

    In Databricks Apps:
    - PGHOST and PGDATABASE are automatically set by the Lakebase resource
    - Service principal credentials are injected

    Locally:
    - Use settings from .env file
    - Use generate_database_credential() for OAuth tokens
    """

    def __init__(self):
        # Check if we're in Databricks Apps by looking for PGHOST
        pghost = os.getenv("PGHOST")

        if pghost:
            # Running in Databricks Apps
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.core import Config

            self._config = Config()
            self._workspace_client = WorkspaceClient()

            self._postgres_username = self._config.client_id
            self._postgres_host = pghost
            self._postgres_database = os.getenv("PGDATABASE", "databricks_postgres")
            self._use_databricks_apps = True

            logger.info(
                "lakebase_factory_initialized",
                host=self._postgres_host,
                database=self._postgres_database,
                auth="databricks_apps_oauth",
            )
        else:
            # Local development - use settings from .env
            from my_app.config import get_settings

            self._use_databricks_apps = False
            settings = get_settings()
            self._postgres_host = settings.lakebase.host
            self._postgres_database = settings.lakebase.database
            self._postgres_username = settings.lakebase.user
            self._local_settings = settings

            logger.info(
                "lakebase_factory_initialized",
                host=self._postgres_host,
                database=self._postgres_database,
                auth="local_oauth",
            )

    def get_connection(self) -> psycopg.Connection:
        """Get a new database connection with fresh OAuth token."""
        if self._use_databricks_apps:
            # Databricks Apps: use service principal OAuth token
            token = self._workspace_client.config.oauth_token().access_token

            return psycopg.connect(
                host=self._postgres_host,
                port=5432,
                dbname=self._postgres_database,
                user=self._postgres_username,
                password=token,
                sslmode="require",
            )
        else:
            # Local development: use generate_database_credential()
            from my_app.config import _token_manager

            token = _token_manager.get_token(
                instance_name=self._local_settings.lakebase.instance_name,
                workspace_host=self._local_settings.databricks.host,
            )

            return psycopg.connect(
                host=self._postgres_host,
                port=5432,
                dbname=self._postgres_database,
                user=self._postgres_username,
                password=token,
                sslmode="require",
            )


# Global connection factory
_factory: LakebaseConnectionFactory | None = None


def get_factory() -> LakebaseConnectionFactory:
    """Get the global connection factory."""
    global _factory
    if _factory is None:
        _factory = LakebaseConnectionFactory()
    return _factory


class PostgresDB:
    """PostgreSQL database client using psycopg with OAuth."""

    def __init__(self):
        self._factory = get_factory()

    @contextmanager
    def session(self) -> Generator[psycopg.Connection, None, None]:
        """Get a database connection context manager."""
        conn = self._factory.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.session() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            return False

    def create_item(self, name: str, description: str | None = None) -> dict:
        """Create a new item."""
        with self.session() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO items (name, description, created_at)
                    VALUES (%s, %s, NOW())
                    RETURNING id, name, description, created_at
                    """,
                    (name, description),
                )
                row = cur.fetchone()
                return {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "created_at": row[3],
                }

    def get_item(self, item_id: int) -> dict | None:
        """Get an item by ID."""
        with self.session() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, description, created_at FROM items WHERE id = %s",
                    (item_id,),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "created_at": row[3],
                    }
                return None


# Global database instance
_db: PostgresDB | None = None


def get_db() -> PostgresDB:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = PostgresDB()
    return _db
```

## FastAPI Application (api/main.py)

```python
"""FastAPI application."""

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from my_app import __version__
from my_app.api.schemas import (
    HealthResponse,
    ItemCreate,
    ItemResponse,
)
from my_app.db.postgres import get_db

logger = structlog.get_logger()

app = FastAPI(
    title="My App API",
    description="API for my Databricks App",
    version=__version__,
)

# CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint with database connectivity status."""
    db = get_db()
    db_status = "connected" if db.health_check() else "disconnected"
    return HealthResponse(
        status="ok",
        version=__version__,
        database=db_status,
    )


@app.post("/api/items", response_model=ItemResponse)
async def create_item(item: ItemCreate) -> ItemResponse:
    """Create a new item."""
    db = get_db()
    result = db.create_item(name=item.name, description=item.description)
    return ItemResponse(**result)


@app.get("/api/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int) -> ItemResponse:
    """Get an item by ID."""
    db = get_db()
    result = db.get_item(item_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Item not found: {item_id}")
    return ItemResponse(**result)
```

## Pydantic Schemas (api/schemas.py)

```python
"""Pydantic request/response schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class ItemCreate(BaseModel):
    """Request body for creating an item."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)


class ItemResponse(BaseModel):
    """Response for an item."""

    id: int
    name: str
    description: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str
    version: str
    database: str
```

## pyproject.toml

```toml
[project]
name = "my-app"
version = "0.1.0"
description = "My Databricks App"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.6.0",
    "databricks-sdk>=0.40.0",
    "psycopg[binary,pool]>=3.2.0",
    "python-dotenv>=1.0.0",
    "structlog>=25.0.0",
    "python-multipart>=0.0.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "httpx>=0.28.0",
]

[build-system]
requires = ["uv_build>=0.9.0,<0.10.0"]
build-backend = "uv_build"

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

## Local Development (.env)

For local development, create a `.env` file:

```bash
# Databricks workspace
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com

# Lakebase connection (for local dev)
LAKEBASE_HOST=your-instance.postgres.azuredatabricks.net
LAKEBASE_DATABASE=databricks_postgres
LAKEBASE_USER=your-service-principal-client-id
LAKEBASE_INSTANCE_NAME=your-lakebase-instance-name

# Logging
LOG_LEVEL=DEBUG
```

## Key Patterns

### Environment Detection
The `LakebaseConnectionFactory` detects Databricks Apps by checking for `PGHOST`:
- **In Databricks Apps**: `PGHOST` and `PGDATABASE` are automatically set by the Lakebase resource
- **Locally**: Uses settings from `.env` file

### OAuth Token Handling
- **In Databricks Apps**: `workspace_client.config.oauth_token().access_token` - automatic refresh
- **Locally**: `w.database.generate_database_credential()` with manual caching in `OAuthTokenManager`

### No Connection Pooling
Each request gets a fresh connection with a fresh OAuth token. This is simpler and works well because:
- OAuth tokens expire after 1 hour
- Connections are lightweight
- Lakebase handles connection management server-side

### Structured Logging
Use `structlog` for JSON-formatted, structured logs:
```python
logger.info("item_created", item_id=item_id, name=name)
logger.error("database_error", error=str(e), item_id=item_id)
```

## Connecting to DBSQL Warehouses

For heavy analytical queries, use `databricks-sql-connector`:

```python
from databricks import sql
from databricks.sdk.core import Config, oauth_service_principal

def get_dbsql_credential_provider():
    """Create OAuth credential provider for DBSQL."""
    config = Config(
        host=f"https://{os.getenv('DATABRICKS_SERVER_HOSTNAME')}",
        client_id=os.getenv("DATABRICKS_CLIENT_ID"),
        client_secret=os.getenv("DATABRICKS_CLIENT_SECRET"),
    )
    return oauth_service_principal(config)

def query_dbsql(query: str, params: list | None = None) -> list[dict]:
    """Execute a query against DBSQL warehouse."""
    with sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        credentials_provider=get_dbsql_credential_provider,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
```

## When to Use Each Database

### Use DBSQL Warehouse for:
- Dashboard queries with aggregations
- Reports spanning large date ranges
- Joins across multiple large tables
- Any query scanning millions+ rows
- Queries using Unity Catalog tables

### Use Lakebase for:
- User session storage
- Application state management
- Real-time CRUD operations
- Low-latency lookups by primary key
- Transactional operations (INSERT, UPDATE, DELETE)
- Data that changes frequently

## Databricks Apps Platform Requirements

### Networking
- Apps must listen on `0.0.0.0` and bind to the port in `DATABRICKS_APP_PORT` environment variable
- Databricks handles TLS termination and requires apps to support HTTP/2 cleartext (H2C)
- Requests route through reverse proxy - don't rely on request origin

### Graceful Shutdown
Apps must terminate within **15 seconds** after receiving SIGTERM, otherwise they are forcibly killed (SIGKILL):

```python
import signal
import sys
from contextlib import asynccontextmanager

import structlog

logger = structlog.get_logger()

# Track shutdown state
_shutdown_requested = False

def handle_sigterm(signum, frame):
    """Handle SIGTERM for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("shutdown_requested", signal=signum)

signal.signal(signal.SIGTERM, handle_sigterm)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with graceful shutdown."""
    logger.info("app_starting")
    yield
    # Cleanup: close DB connections, finish pending work
    logger.info("app_shutting_down")
    # Must complete within 15 seconds!

app = FastAPI(lifespan=lifespan)
```

### System Limitations
- **No system packages**: Cannot use apt-get, yum, apk - use PyPI/npm for dependencies only
- Individual files cannot exceed 10 MB
- App logs are deleted when compute terminates
- Only accessible to authenticated Databricks users (no public access)

## Global Exception Handling

Implement global exception handlers to prevent crashes and avoid exposing stack traces:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import structlog

logger = structlog.get_logger()

app = FastAPI()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    # Never expose internal details to clients
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    logger.warning("validation_error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )
```

## Caching for Performance

Use in-memory caching for expensive operations:

```python
from functools import lru_cache
from cachetools import TTLCache
import structlog

logger = structlog.get_logger()

# Simple LRU cache for pure functions
@lru_cache(maxsize=100)
def get_config_value(key: str) -> str:
    """Cache configuration lookups."""
    return expensive_config_fetch(key)

# TTL cache for data that expires
_item_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)  # 5 min TTL

def get_item_cached(item_id: int) -> dict | None:
    """Get item with caching."""
    if item_id in _item_cache:
        logger.debug("cache_hit", item_id=item_id)
        return _item_cache[item_id]

    logger.debug("cache_miss", item_id=item_id)
    item = db.get_item(item_id)
    if item:
        _item_cache[item_id] = item
    return item
```

Add `cachetools` to dependencies:
```toml
dependencies = [
    # ... existing deps
    "cachetools>=5.0.0",
]
```

## Long-Running Operations

Use async patterns for operations that may timeout. Start the operation, return immediately, then poll for status:

```python
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class JobResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    result: dict | None = None
    error: str | None = None

# In-memory job tracking (use Lakebase for persistence)
_jobs: dict[UUID, JobResponse] = {}

@app.post("/api/jobs", response_model=JobResponse)
async def create_job(request: JobRequest) -> JobResponse:
    """Start a long-running job."""
    job_id = uuid4()
    job = JobResponse(job_id=job_id, status=JobStatus.PENDING)
    _jobs[job_id] = job

    # Start background task (or trigger Databricks Job)
    asyncio.create_task(run_job_async(job_id, request))

    return job

@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: UUID) -> JobResponse:
    """Poll job status."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]
```

## Security Best Practices

### Access Control
- Grant minimum necessary permissions: prefer `CAN USE` over `CAN MANAGE`
- Use **dedicated service principal for each app** - never share across apps
- Separate dev/staging/prod apps across different workspaces

### Secrets Management
- Never expose raw secret values in environment variables
- Use `valueFrom` in app.yaml to reference secrets:
  ```yaml
  env:
    - name: API_KEY
      valueFrom: my-secret
  ```
- Rotate secrets regularly when team roles change

### SQL Injection Prevention
Always use parameterized queries - never string concatenation:

```python
# CORRECT - parameterized query
cur.execute(
    "SELECT * FROM items WHERE id = %s AND status = %s",
    (item_id, status),
)

# WRONG - SQL injection vulnerability
cur.execute(f"SELECT * FROM items WHERE id = {item_id}")  # NEVER DO THIS
```

### Input Validation
Pydantic handles validation, but add explicit checks for business logic:

```python
from pydantic import BaseModel, Field, field_validator

class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    quantity: int = Field(..., ge=0, le=10000)

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        # Remove potentially dangerous characters
        return v.strip().replace("<", "").replace(">", "")
```

### Audit Logging
Log all significant user actions:

```python
@app.post("/api/items")
async def create_item(item: ItemCreate, request: Request) -> ItemResponse:
    user = get_current_user(request)
    result = db.create_item(item.name, item.description)

    logger.info(
        "item_created",
        item_id=result["id"],
        user_email=user.email,
        action="create",
        resource_type="item",
    )
    return ItemResponse(**result)
```

## Architecture: Offload Heavy Processing

App compute is optimized for UI rendering. Offload heavy workloads to Databricks services:

| Workload | Use This Service |
|----------|------------------|
| Complex SQL queries | DBSQL Warehouse |
| ML inference | Model Serving endpoints |
| ETL/batch processing | Databricks Jobs |
| Large data processing | Spark clusters |

```python
# Example: Trigger a Databricks Job for heavy processing
from databricks.sdk import WorkspaceClient

def trigger_etl_job(input_path: str) -> int:
    """Trigger ETL job and return run ID."""
    w = WorkspaceClient()
    run = w.jobs.run_now(
        job_id=123,
        notebook_params={"input_path": input_path},
    )
    return run.run_id

def check_job_status(run_id: int) -> str:
    """Check job run status."""
    w = WorkspaceClient()
    run = w.jobs.get_run(run_id)
    return run.state.life_cycle_state.value
```
