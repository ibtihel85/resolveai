# Dockerfile
#
# Production container for the ResolveAI API.
# Runs the FastAPI application via uvicorn.
#
# Build:  docker build -t resolveai .
# Run:    docker compose up
#
# The analytics dashboard runs as a separate container
# defined in docker-compose.yml (Dockerfile.dashboard).

# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim — Debian-based, minimal footprint (~150MB vs ~900MB full)
# Excludes development tools not needed at runtime.
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential: required to compile some Python packages (psycopg2, etc.)
# curl: used in healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
# rm -rf /var/lib/apt/lists/* removes the apt cache — keeps image size small

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────────────────
# Copy only pyproject.toml first — Docker caches this layer.
# If only source code changes (not dependencies), this layer is reused,
# making rebuilds much faster.
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

# ── Download spaCy model for Presidio PII detection ──────────────────────────
# Must be done after pip install, before copying source code.
# Cached in its own layer — only re-downloaded when dependencies change.
RUN python -m spacy download en_core_web_lg

# ── Copy source code ──────────────────────────────────────────────────────────
# Done last — source code changes most frequently.
# Putting it last maximises Docker layer cache hits.
COPY src/ ./src/
COPY scripts/ ./scripts/

# ── Runtime configuration ─────────────────────────────────────────────────────
# Port the FastAPI app listens on inside the container
EXPOSE 8000

# Healthcheck — Docker monitors this endpoint to know if the container is healthy
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start command ─────────────────────────────────────────────────────────────
# --host 0.0.0.0: listen on all interfaces (required inside Docker)
# --port 8000: internal port (mapped to host port in docker-compose.yml)
# No --reload in production: reload is for development only
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]