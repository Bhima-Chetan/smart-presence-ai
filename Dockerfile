# Use Python 3.10 slim image for better compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for face recognition and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libdlib-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads static/exports training_data models

# Set permissions
RUN chmod +x app.py

# Expose port
EXPOSE $PORT

# Run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120"]