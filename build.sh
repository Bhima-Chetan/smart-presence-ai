#!/bin/bash

# Build script for Render deployment
echo "Starting build process for Smart Presence AI..."

# Update system packages
echo "Installing system dependencies..."

# Install Python dependencies
echo "Installing Python packages..."
pip install --upgrade pip setuptools wheel

# Install CMake first
pip install cmake

# Install numpy before other packages
pip install numpy==1.24.3

# Install computer vision packages
pip install opencv-python-headless==4.8.1.78

# Try to install dlib and face recognition
echo "Installing face recognition libraries..."
pip install dlib==19.24.2 || echo "dlib installation failed, will use fallback"
pip install face-recognition==1.3.0 || echo "face-recognition installation failed, will use fallback"

# Install remaining requirements
pip install -r requirements.txt || echo "Some packages may have failed, continuing..."

echo "Build process completed!"
