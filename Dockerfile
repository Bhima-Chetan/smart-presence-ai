# Use Python 3.10 with pre-installed build tools
FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies efficiently
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with specific order to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir cmake
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir dlib --no-build-isolation
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads static/exports training_data models

# Expose port
EXPOSE 8080

# Run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120"]