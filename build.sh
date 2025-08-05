#!/bin/bash

# Simplified build script for Render deployment
echo "Starting simplified build process for Smart Presence AI..."

# Update package lists
apt-get update

# Install system CMake (not Python CMake)
echo "Installing system dependencies..."
apt-get install -y cmake build-essential

# Install Python dependencies with explicit pip upgrade
echo "Installing Python packages..."
python -m pip install --upgrade pip setuptools wheel

# Install core packages first (these should always work)
echo "Installing core Flask dependencies..."
pip install flask==2.3.3
pip install gunicorn==21.2.0
pip install werkzeug==2.3.7

# Install data processing packages
echo "Installing data processing packages..."
pip install pandas==2.0.3
pip install pillow==10.0.0

# Install NumPy with specific version (critical)
echo "Installing NumPy 1.24.3..."
pip install numpy==1.24.3

# Install OpenCV (should work on most systems)
echo "Installing OpenCV..."
pip install opencv-python-headless==4.8.1.78

# Skip face recognition libraries entirely for now
echo "Skipping face recognition libraries due to build complexity..."
echo "Application will use ultra-simple processor mode"

# Install any remaining simple packages
echo "Installing remaining requirements..."
pip install python-dateutil==2.8.2 || echo "Some packages failed, continuing..."

echo "Build completed! Application will run in simple mode without face recognition."
echo "Final package list:"
pip list
