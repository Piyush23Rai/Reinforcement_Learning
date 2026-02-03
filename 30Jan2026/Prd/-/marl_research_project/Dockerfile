# ═══════════════════════════════════════════════════════════════════════════════
# MARL RESEARCH DOCKER IMAGE
# Complete environment for Multi-Agent Reinforcement Learning research
# ═══════════════════════════════════════════════════════════════════════════════

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

LABEL maintainer="MARL Research"
LABEL description="Multi-Agent Reinforcement Learning research environment"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional MARL libraries
RUN pip install --no-cache-dir \
    pettingzoo[all] \
    supersuit \
    gymnasium \
    agilerl \
    wandb \
    tensorboard \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml

# Copy project files
COPY . .

# Create directories for outputs
RUN mkdir -p /app/results /app/models /app/logs

# Set up volumes for persistent storage
VOLUME ["/app/results", "/app/models", "/app/logs"]

# Expose TensorBoard port
EXPOSE 6006

# Default command
CMD ["python", "scripts/train_maddpg.py", "--env", "simple_spread", "--episodes", "5000"]
