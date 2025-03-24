# Containerization: Docker with Multi-Stage Builds for ML

## Introduction

Docker containerization has become essential in modern machine learning workflows, allowing for consistent development, testing, and deployment environments. Multi-stage builds are particularly valuable for ML applications, as they help create smaller, more secure production images while maintaining all the tools needed during development.

## Benefits of Multi-Stage Builds for ML

1. **Reduced image size**: Final containers include only runtime dependencies
2. **Improved security**: Fewer packages means smaller attack surface
3. **Faster deployments**: Smaller images are quicker to transfer and start
4. **Separation of concerns**: Build tools in one stage, runtime in another
5. **Dependency management**: Clear isolation of development vs. production dependencies

## Multi-Stage Build Example for ML Project

Below is a template Dockerfile that demonstrates multi-stage builds for a typical machine learning project using Python:

```dockerfile
# ===== Build Stage =====
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# ===== Runtime Stage =====
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Create non-root user for better security
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Copy only wheels from builder stage
COPY --from=builder /app/wheels /app/wheels
COPY --from=builder /app/requirements.txt .

# Install dependencies from wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --no-index --find-links=/app/wheels -r requirements.txt && \
    rm -rf /app/wheels

# Copy application code
COPY --chown=appuser:appuser ./src/ /app/src/
COPY --chown=appuser:appuser ./models/ /app/models/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model.pkl

# Command to run on container start
ENTRYPOINT ["python", "/app/src/serve.py"]
```

## Advanced Multi-Stage Techniques for ML

### 1. GPU Support with CUDA

```dockerfile
# Build stage
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Install Python and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev \
    build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*

# Install ML dependencies
COPY requirements.txt .
RUN pip3 wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Runtime stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /app/wheels /app/wheels
COPY requirements.txt .
RUN pip3 install --no-cache-dir --no-index --find-links=/app/wheels -r requirements.txt

# Copy model and code
COPY ./app /app
WORKDIR /app

# Runtime command
ENTRYPOINT ["python3", "serve.py"]
```

### 2. Model Training and Serving Separation

```dockerfile
# Base stage with common dependencies
FROM python:3.10-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements-common.txt .
RUN pip install --no-cache-dir -r requirements-common.txt

# Training stage
FROM base AS training
COPY requirements-training.txt .
RUN pip install --no-cache-dir -r requirements-training.txt
COPY ./training /app/training
WORKDIR /app
ENTRYPOINT ["python", "/app/training/train.py"]

# Inference stage
FROM base AS inference
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt
COPY ./inference /app/inference
COPY --from=training /app/models/model.pkl /app/models/model.pkl
WORKDIR /app
ENTRYPOINT ["python", "/app/inference/serve.py"]
```

## Best Practices

1. **Use specific base images**: Tag specific versions to ensure consistency
2. **Leverage the build cache**: Order Dockerfile commands from least to most likely to change
3. **Minimize layers**: Combine related commands with `&&` to reduce image size
4. **Non-root users**: Run containers with limited permissions for security
5. **Leverage .dockerignore**: Exclude unnecessary files from the build context
6. **Environment variables**: Use ENV for configuration to avoid hardcoded values
7. **Health checks**: Add HEALTHCHECK instructions for better orchestration
8. **Optimize for caching**: Copy requirements files first, before copying application code

## Example .dockerignore File

```
# Version control
.git
.gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Development and IDE files
.idea/
.vscode/
*.swp
*.swo

# OS specific
.DS_Store
Thumbs.db

# Data and models (if stored elsewhere)
data/
models/
*.h5
*.pkl
*.onnx

# Documentation
docs/
README.md
LICENSE

# Docker files themselves
Dockerfile*
docker-compose*
```

## Docker Compose for ML Development

```yaml
version: '3'

services:
  training:
    build:
      context: .
      dockerfile: Dockerfile
      target: training
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
      - EPOCHS=100
      - BATCH_SIZE=32
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  inference:
    build:
      context: .
      dockerfile: Dockerfile
      target: inference
    volumes:
      - ./models:/app/models
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/model.pkl
      - PORT=8000
    depends_on:
      - training
```

## Conclusion

Docker multi-stage builds offer significant advantages for machine learning projects, including reduced image sizes, improved security, and clearer separation of development and production concerns. By implementing the techniques and best practices outlined in this guide, you can create more efficient, reproducible, and deployable ML solutions with Docker.
