# Multi-stage Dockerfile for chaos-rng development and production

# Development stage
FROM python:3.11-slim as development

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements-dev.txt .
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .[all]

# Install pre-commit hooks
RUN pre-commit install

# Production stage
FROM python:3.11-slim as production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (minimal for production)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install only production dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip

# Copy source code
COPY src/ src/
COPY README.md LICENSE ./

# Install package
RUN pip install .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Testing stage
FROM development as testing

# Run tests by default
CMD ["make", "test"]

# Linting stage  
FROM development as linting

# Run linting by default
CMD ["make", "lint"]

# Documentation stage
FROM development as docs

# Install additional docs dependencies
RUN pip install sphinx-autobuild

# Expose port for docs server
EXPOSE 8000

# Run docs server by default
CMD ["make", "docs-live"]
