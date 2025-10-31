# ============================================
# CORE CARE - Multi-stage Dockerfile
# Optimized for ML/GPU workloads
# ============================================

# ============================================
# Stage 1: Base Image with CUDA Support
# ============================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ============================================
# Stage 2: Dependencies Installation
# ============================================
FROM base AS dependencies

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install PyTorch with CUDA support first (largest dependency)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 3: Application
# ============================================
FROM dependencies AS application

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_CACHE=/app/cache/transformers \
    HF_HOME=/app/cache/huggingface \
    MPLCONFIGDIR=/app/cache/matplotlib

# Create necessary directories
RUN mkdir -p /app/cache/transformers \
    /app/cache/huggingface \
    /app/cache/matplotlib \
    /app/logs \
    /app/uploads \
    /app/temp

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run application
CMD ["python", "run.py"]

# ============================================
# Alternative: Production with Gunicorn
# ============================================
# Uncomment for production deployment:
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", \
#      "--workers", "2", \
#      "--threads", "4", \
#      "--timeout", "120", \
#      "--worker-class", "sync", \
#      "--log-level", "info", \
#      "run:app"]