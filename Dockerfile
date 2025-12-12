FROM python:3.10-slim

# System deps (kept minimal).
# - git: sometimes needed by HF tooling
# - libgomp1: required by many torch wheels for OpenMP
# - ca-certificates: required for HTTPS downloads (HF models/datasets) in slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the project
COPY . /app

# Recommended runtime env vars (override as needed)
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

# Default command shows usage; override in docker run
CMD ["python", "evaluate_models.py", "--help"]


