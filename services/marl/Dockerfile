FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application (core code and models are mounted as volumes via docker-compose)
COPY app/ /app/

# Environment
ENV MODEL_DIR=/models/marl
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8003/health')"

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
