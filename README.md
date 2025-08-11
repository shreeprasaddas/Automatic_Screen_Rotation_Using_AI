# Automatic Screen Rotation Using AI

This project automatically rotates your screen based on face orientation using AI. It combines YOLOv5 for face detection and a custom CNN model trained on the BIWI dataset for head pose estimation.

## � Features
- Real-time face detection using YOLOv5
- Head pose estimation (yaw, pitch, roll)
- Automatic screen rotation based on face orientation
- Cross-platform support (Windows and Linux)
- Easy-to-use interface
- Visual feedback with real-time angle display
- Support for both automatic and manual rotation
- Optimized for performance with GPU acceleration

## 📁 Project Structure
```
├── main_simple_rotation.py    # Main script for Windows
├── linux_screen_rotation_simple.py  # Main script for Linux
├── scripts/
│   ├── data_loader.py        # Dataset loading utilities
│   ├── predict_pose.py       # Head pose prediction
│   ├── screen_rotator.py     # Screen rotation logic
│   ├── train_pose_model.py   # Model training script
│   └── utils.py             # Utility functions
├── models/
│   ├── head_pose_cnn.pth    # Head pose model weights
│   └── yolov5_face.pt       # YOLOv5 face detection model
└── requirements.txt         # Python dependencies
```

## �️ System Requirements
- Python 3.10 or higher
- Webcam
- NVIDIA GPU (recommended for better performance)
- Operating System: Windows 10/11 or Linux

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/shreeprasaddas/Automatic_Screen_Rotation_Using_AI.git
cd Automatic_Screen_Rotation_Using_AI
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download model weights:
- Place YOLOv5 face detection model (`yolov5_face.pt`) in the `models` folder
- Train or download head pose estimation model (`head_pose_cnn.pth`) to the `models` folder

## 🚀 Usage

### For Windows Users:
```bash
python main_simple_rotation.py
```

### For Linux Users:
```bash
python linux_screen_rotation_simple.py
```

### Training Custom Model (Optional):
If you want to train your own head pose model:
```bash
python scripts/train_pose_model.py
```

## 🎮 Controls

- **'q'**: Quit the application
- **'m'**: Test manual rotation (cycles through all rotations)
- **'f'**: Show face detection statistics
- **'s'**: Show current settings
- **'t'**: Show threshold adjustment info

## ⚙️ Configuration
- Default rotation thresholds:
  - Pitch > 25° : Portrait mode
  - Pitch < -25° : Reverse portrait
  - Yaw > 25° : Landscape
  - Yaw < -25° : Reverse landscape

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## � License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- YOLOv5 Face Detection from [yolov5-face](https://github.com/deepcam-cn/yolov5-face)
- BIWI Dataset for head pose estimation