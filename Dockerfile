FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including TA-Lib requirements)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib (required for technical analysis)
# Download updated config.guess/config.sub for ARM compatibility
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    wget -O config.guess 'https://git.savannah.gnu.org/cgit/config.git/plain/config.guess' && \
    wget -O config.sub 'https://git.savannah.gnu.org/cgit/config.git/plain/config.sub' && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
COPY scripts/ scripts/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/lib

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

# Expose port
EXPOSE 8050

# Default command
CMD ["python", "-m", "src.web.app"]

