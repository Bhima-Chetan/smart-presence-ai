#!/bin/bash

# Build script for Render deployment
echo "Starting build process for Smart Presence AI..."

# Set environment variables for better compilation
export MAKEFLAGS="-j1"
export CMAKE_BUILD_PARALLEL_LEVEL=1

# Update system packages and install dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev

# Install Python dependencies
echo "Installing Python packages..."
pip install --upgrade pip setuptools wheel

# Install CMake and build tools first
pip install cmake

# Install numpy with specific version (critical for compatibility)
echo "Installing NumPy..."
pip install numpy==1.24.3

# Install computer vision packages
echo "Installing OpenCV..."
pip install opencv-python-headless==4.8.1.78

# Try to install dlib with timeout and fallback
echo "Attempting to install dlib..."
timeout 600 pip install dlib==19.24.2 || {
    echo "dlib installation failed or timed out, trying alternative..."
    pip install dlib || echo "dlib installation failed completely, will use fallback processor"
}

# Try to install face recognition with fallback
echo "Attempting to install face-recognition..."
pip install face-recognition==1.3.0 || {
    echo "face-recognition installation failed, will use simplified processor"
}

# Install core Flask dependencies (these should always work)
echo "Installing core dependencies..."
pip install flask==2.3.3
pip install gunicorn==21.2.0
pip install pandas==2.0.3
pip install Pillow==10.0.0
pip install werkzeug==2.3.7

# Try to install remaining requirements, continue on errors
echo "Installing remaining requirements..."
pip install -r requirements.txt || echo "Some optional packages failed, continuing with available packages..."

echo "Build process completed successfully!"
echo "Installed packages:"
pip list | grep -E "(flask|gunicorn|numpy|opencv|dlib|face-recognition)"
