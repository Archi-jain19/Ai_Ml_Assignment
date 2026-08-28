# ==============================================================================
# Backend Dockerfile for FacetLens Scoring Pipeline
# Python 3.12 Slim, Non-Root User, Persistent HuggingFace Cache
# ==============================================================================

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    PORT=8000

# Install runtime and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root user and persistent cache directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/.cache/huggingface /app/.cache/sentence_transformers /app/data/processed /app/outputs && \
    chown -R appuser:appuser /app

# Pre-install CPU-optimized PyTorch wheel (avoids multi-GB CUDA bloat)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements for remaining dependencies
COPY requirements.txt .

# Install remaining Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser tests/ ./tests/
COPY --chown=appuser:appuser server.py .

# Switch to non-root user
USER appuser

# Expose backend API port
EXPOSE 8000

# Health check to monitor backend availability without expensive model loads
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production start command
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
