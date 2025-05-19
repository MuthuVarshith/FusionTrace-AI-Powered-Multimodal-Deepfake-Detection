FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (including ffmpeg for pydub)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy pyproject.toml and uv.lock for dependency installation
COPY backend/pyproject.toml .
COPY backend/uv.lock .

# Install dependencies using uv
RUN uv sync --no-cache

# Copy the backend code
COPY backend/app ./app
COPY backend/models ./models
COPY backend/templates ./templates
COPY backend/config.yaml .

# Create directory for updated data
RUN mkdir -p /app/data

# Expose the port the app runs on (matches Lightning AI URL)
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]