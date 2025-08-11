#!/bin/bash

echo "🚀 Installing Dependencies for Auto Screen Rotation System"
echo "=================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (CPU version for compatibility)
echo "🤖 Installing PyTorch (CPU version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

# Install additional system dependencies for Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Installing Linux system dependencies..."
    
    # Check if we're on Ubuntu/Debian
    if command -v apt-get &> /dev/null; then
        echo "📦 Installing system packages via apt..."
        sudo apt-get update
        sudo apt-get install -y \
            python3-dev \
            python3-pip \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1 \
            libgthread-2.0-0 \
            libgtk-3-0 \
            libavcodec-dev \
            libavformat-dev \
            libswscale-dev \
            libv4l-dev \
            libxvidcore-dev \
            libx264-dev \
            libjpeg-dev \
            libpng-dev \
            libtiff-dev \
            libatlas-base-dev \
            gfortran \
            libhdf5-dev \
            libhdf5-serial-dev \
            libhdf5-103 \
            libqtgui4 \
            libqtwebkit4 \
            libqt4-test \
            python3-pyqt5 \
            libgstreamer1.0-0 \
            libgstreamer-plugins-base1.0-0 \
            libgtk2.0-dev \
            libtbb-dev \
            libtbb2 \
            libdc1394-22-dev \
            libdc1394-22 \
            libdc1394-dev \
            libdc1394-22-dev \
            libdc1394-22 \
            libdc1394-dev
    elif command -v yum &> /dev/null; then
        echo "📦 Installing system packages via yum..."
        sudo yum install -y \
            python3-devel \
            python3-pip \
            mesa-libGL \
            glib2-devel \
            libSM \
            libXext \
            libXrender-devel \
            libgomp \
            gtk3-devel \
            ffmpeg-devel \
            libjpeg-devel \
            libpng-devel \
            libtiff-devel \
            atlas-devel \
            gcc-gfortran \
            hdf5-devel \
            qt4-devel \
            gstreamer1-devel \
            gstreamer1-plugins-base-devel \
            tbb-devel \
            libdc1394-devel
    elif command -v pacman &> /dev/null; then
        echo "📦 Installing system packages via pacman..."
        sudo pacman -S --needed \
            python-pip \
            python-setuptools \
            python-wheel \
            opencv \
            hdf5 \
            qt4 \
            gstreamer \
            gst-plugins-base \
            tbb \
            libdc1394
    else
        echo "⚠️  Unknown package manager. Please install system dependencies manually."
    fi
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "📁 Creating models directory..."
    mkdir -p models
    echo "✅ Models directory created"
fi

# Check if required model files exist
echo "🔍 Checking for required model files..."

if [ ! -f "models/yolov5_face.pt" ]; then
    echo "⚠️  Warning: models/yolov5_face.pt not found"
    echo "   You may need to download or train this model"
fi

if [ ! -f "models/head_pose_cnn.pth" ]; then
    echo "⚠️  Warning: models/head_pose_cnn.pth not found"
    echo "   You may need to download or train this model"
fi

if [ ! -f "models/best_rotation_model.pth" ]; then
    echo "⚠️  Warning: models/best_rotation_model.pth not found"
    echo "   You may need to download or train this model"
fi

# Create templates directory for Flask
if [ ! -d "templates" ]; then
    echo "📁 Creating templates directory for Flask..."
    mkdir -p templates
    echo "✅ Templates directory created"
fi

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
except ImportError as e:
    print(f'❌ PyTorch: {e}')

try:
    import torchvision
    print(f'✅ TorchVision: {torchvision.__version__}')
except ImportError as e:
    print(f'❌ TorchVision: {e}')

try:
    import cv2
    print(f'✅ OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'❌ OpenCV: {e}')

try:
    import flask
    print(f'✅ Flask: {flask.__version__}')
except ImportError as e:
    print(f'❌ Flask: {e}')

try:
    import flask_socketio
    print(f'✅ Flask-SocketIO: {flask_socketio.__version__}')
except ImportError as e:
    print(f'❌ Flask-SocketIO: {e}')

try:
    import numpy
    print(f'✅ NumPy: {numpy.__version__}')
except ImportError as e:
    print(f'❌ NumPy: {e}')

try:
    import ultralytics
    print(f'✅ Ultralytics: {ultralytics.__version__}')
except ImportError as e:
    print(f'❌ Ultralytics: {e}')
"

echo ""
echo "🎉 Installation Complete!"
echo "========================"
echo ""
echo "📋 Next Steps:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Run web server: python web_rotation_server.py"
echo "3. Open browser to: http://localhost:5000"
echo ""
echo "🔧 Alternative commands:"
echo "- Simple rotation: python main_simple_rotation.py"
echo "- Linux version: python linux_screen_rotation.py"
echo ""
echo "⚠️  Note: You may need to download model files if they don't exist"
echo "   - models/yolov5_face.pt (YOLOv5 face detection)"
echo "   - models/head_pose_cnn.pth (Head pose estimation)"
echo "   - models/best_rotation_model.pth (Rotation prediction)"
echo ""
echo "🚀 Happy coding!"
