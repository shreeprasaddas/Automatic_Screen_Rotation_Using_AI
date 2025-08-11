# 🚀 Installation Guide for Auto Screen Rotation System

This guide will help you install all dependencies and set up the Auto Screen Rotation System on your machine.

## 📋 Prerequisites

- **Python 3.8+** (Python 3.9 or 3.10 recommended)
- **pip** (Python package installer)
- **Git** (for cloning the repository)
- **Administrator/sudo access** (for system dependencies)

## 🎯 Quick Installation

### Option 1: Automated Installation (Recommended)

#### For Linux/macOS:
```bash
# Make the script executable
chmod +x install_dependencies.sh

# Run the installation script
./install_dependencies.sh
```

#### For Windows:
```bash
# Run the Python installation script
python install_dependencies.py
```

### Option 2: Manual Installation

#### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

#### Step 2: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for compatibility)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt
```

## 🔧 System-Specific Dependencies

### Linux (Ubuntu/Debian)
```bash
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
    libhdf5-serial-dev
```

### Linux (CentOS/RHEL/Fedora)
```bash
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
    hdf5-devel
```

### Linux (Arch Linux)
```bash
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
```

### Windows
- No additional system dependencies required
- Ensure you have the latest graphics drivers installed

### macOS
- No additional system dependencies required
- Ensure you have Xcode command line tools: `xcode-select --install`

## 📦 Required Python Packages

The system requires the following Python packages:

### Core AI and Computer Vision
- **torch** >= 2.0.0 - PyTorch deep learning framework
- **torchvision** >= 0.15.0 - Computer vision models and transforms
- **opencv-python** >= 4.8.0 - Computer vision library
- **numpy** >= 1.21.0 - Numerical computing
- **pillow** >= 9.0.0 - Image processing

### Web Framework
- **flask** >= 2.3.0 - Web framework
- **flask-socketio** >= 5.3.0 - WebSocket support
- **python-socketio** >= 5.8.0 - Socket.IO implementation

### Data Processing and Visualization
- **matplotlib** >= 3.5.0 - Plotting library
- **scipy** >= 1.9.0 - Scientific computing
- **pandas** >= 1.5.0 - Data manipulation
- **seaborn** >= 0.11.0 - Statistical visualization

### Utilities
- **pyyaml** >= 6.0 - YAML parser
- **tqdm** >= 4.64.0 - Progress bars
- **requests** >= 2.28.0 - HTTP library

### YOLOv5 Dependencies
- **ultralytics** >= 8.0.0 - YOLOv5 implementation

## 🎮 GPU Support (Optional)

If you have a CUDA-capable GPU and want to use it:

### CUDA 11.8
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Note**: CPU version is recommended for compatibility and easier setup.

## 📁 Directory Structure

After installation, you should have:

```
auto_screen_rotation/
├── .venv/                          # Virtual environment
├── models/                         # AI model files
├── templates/                      # Flask HTML templates
├── dataset/                        # Training dataset
├── scripts/                        # Core utilities
├── requirements.txt                # Python dependencies
├── install_dependencies.sh         # Linux/macOS installer
├── install_dependencies.py         # Python installer
└── [other Python files]
```

## 🔍 Model Files

The system requires these pre-trained models:

- **`models/yolov5_face.pt`** - YOLOv5 face detection model
- **`models/head_pose_cnn.pth`** - Head pose estimation CNN
- **`models/best_rotation_model.pth`** - Screen rotation prediction model

**Note**: These models may not be included in the repository. You may need to:
1. Download pre-trained models
2. Train your own models
3. Use alternative models

## 🧪 Testing Installation

After installation, test if everything works:

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Test imports
python -c "
import torch
import torchvision
import cv2
import flask
import flask_socketio
import numpy
import ultralytics
print('✅ All packages imported successfully!')
"
```

## 🚀 Running the System

### Web Interface (Recommended)
```bash
python web_rotation_server.py
# Open browser to: http://localhost:5000
```

### Simple Rotation System
```bash
python main_simple_rotation.py
```

### Linux-Specific Version
```bash
python linux_screen_rotation.py
```

## ❌ Troubleshooting

### Common Issues

#### 1. **Import Errors**
- Ensure virtual environment is activated
- Check if all packages are installed: `pip list`
- Reinstall packages: `pip install -r requirements.txt`

#### 2. **OpenCV Issues (Linux)**
- Install system dependencies: `sudo apt-get install libgl1-mesa-glx`
- Reinstall OpenCV: `pip uninstall opencv-python && pip install opencv-python`

#### 3. **PyTorch Issues**
- Use CPU version for compatibility: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- Check PyTorch version compatibility

#### 4. **Permission Issues**
- Run as administrator (Windows) or with sudo (Linux)
- Check file permissions: `chmod +x install_dependencies.sh`

#### 5. **Model Loading Errors**
- Ensure model files exist in `models/` directory
- Check file paths and permissions
- Download or train required models

### Getting Help

1. **Check error messages** carefully
2. **Verify Python version**: `python --version`
3. **Check virtual environment**: `which python`
4. **List installed packages**: `pip list`
5. **Check system dependencies** are installed

## 📚 Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **OpenCV Documentation**: https://docs.opencv.org/
- **Flask Documentation**: https://flask.palletsprojects.com/
- **YOLOv5 Documentation**: https://docs.ultralytics.com/

## 🎉 Success!

Once installation is complete, you can:

1. **Start the web interface** for a full-featured experience
2. **Run simple rotation** for basic functionality
3. **Customize thresholds** and settings
4. **Train your own models** for better accuracy
5. **Extend the system** with new features

Happy coding! 🚀
