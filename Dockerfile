FROM python:3.12-slim

LABEL maintainer="VK-Ant (Venkatkumar Rajan)"
LABEL description="sonarwise — Pluggable audio perception engine. Hear. Search. Retrieve."
LABEL version="0.1.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy project
COPY pyproject.toml README.md LICENSE ./
COPY sonarwise/ sonarwise/
COPY tests/ tests/
COPY assets/ assets/

# Install sonarwise with all dependencies
# Use --break-system-packages since we're in a container
RUN pip install --no-cache-dir ".[all,dev]" 2>/dev/null || \
    pip install --no-cache-dir ".[dev]"

# Default data directory
RUN mkdir -p /data

# Expose data volume
VOLUME ["/data"]

# Default command — show help
ENTRYPOINT ["sonarwise"]
CMD ["--help"]
