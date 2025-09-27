FROM python:3.13 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Create virtualenv early for cached layer of dependencies
RUN python -m venv .venv

# Copy minimal build context first (metadata + package source) for better cache reuse
COPY pyproject.toml README.md ./
COPY app ./app

# Install project (deps + package) into venv
RUN .venv/bin/pip install --no-cache-dir .
FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .
# Run with uvicorn directly (more stable than fastapi CLI wrapper)
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
