# syntax=docker/dockerfile:1

# Use the uv Python base image for fast dependency installs
FROM ghcr.io/astral-sh/uv:python3.11-bookworm

# Create working directory
WORKDIR /app

# Copy project metadata first for dependency resolution and caching
COPY pyproject.toml ./
COPY config.toml ./config.toml

# Install project dependencies into the environment
RUN uv sync --no-dev

# Copy the rest of the application source
COPY src ./src

# Expose the FastAPI default port
EXPOSE 8000

# Run the FastAPI application with uvicorn via uv
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
