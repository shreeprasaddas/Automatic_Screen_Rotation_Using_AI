# Automatic Screen Rotation System

A real-time face detection and head pose estimation system that provides intelligent screen rotation recommendations based on head orientation.

## 🎯 Features

- **Real-time Face Detection**: Uses YOLOv5 with optimized settings for accurate face detection
- **Head Pose Estimation**: CNN model predicts yaw, pitch, and roll angles
- **Smart Rotation Logic**: Determines optimal screen rotation based on head tilt
- **Manual Rotation Guidance**: Provides step-by-step instructions for manual rotation
- **Visual Feedback**: Real-time display of pose angles and recommendations
- **Optimized Performance**: Enhanced preprocessing and fallback model support

## 📁 Project Structure

```
auto_screen_rotation/
├── main_final.py                    # Final optimized system (RECOMMENDED)
├── main_with_manual_rotation.py     # Manual rotation system
├── main.py                         # Original system (automatic rotation)
├── scripts/                        # Core components
│   ├── predict_pose.py             # Head pose estimation
│   ├── utils.py                    # Screen rotation utilities
│   ├── data_loader.py              # BIWI dataset loader
│   └── train_pose_model.py         # Model training script
├── models/                         # Pre-trained models
│   ├── head_pose_cnn.pth           # Head pose CNN model
│   └── yolov5_face.pt              # YOLOv5 face detection model
├── dataset/                        # BIWI dataset (for training)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── SOLUTION_SUMMARY.md             # Screen rotation issue solution
└── SCREEN_ROTATION_TROUBLESHOOTING.md # Troubleshooting guide
```

## 🚀 Quick Start

### Prerequisites
- Windows 10/11
- Python 3.8+
- Webcam
- Virtual environment (recommended)

### Installation

1. **Clone or download the project**
2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the optimized system:**
   ```bash
   python main_final.py
   ```

## 🎮 Controls

- **'q'**: Quit the application
- **'m'**: Test manual rotation (cycles through all rotations)
- **'f'**: Show face detection statistics
- **'s'**: Show current settings
- **'t'**: Show threshold adjustment info

## ⚙️ Configuration

### Face Detection Settings
- **Confidence Threshold**: 0.25 (adjustable in code)
- **Model**: Custom YOLOv5 face model with fallback options
- **Frame Size**: 1280x720
- **FPS**: 30

### Rotation Thresholds
- **Roll Threshold**: 15° (head tilt left/right)
- **Pitch Threshold**: 12° (head up/down)
- **Lower values**: More sensitive to head movement
- **Higher values**: Less sensitive to head movement

## 🔧 Troubleshooting

### Face Detection Issues
1. **No face detected**: 
   - Ensure good lighting
   - Face the camera directly
   - Check webcam permissions
   - Try adjusting confidence threshold

2. **Poor detection quality**:
   - Clean webcam lens
   - Improve lighting conditions
   - Move closer to camera
   - Check for background interference

### Screen Rotation Issues
- **Manual rotation only**: Your system doesn't support automatic rotation
- **Follow on-screen instructions**: Use Windows display settings
- **Keyboard shortcuts**: Try Ctrl + Alt + arrow keys

## 📊 Performance

The optimized system provides:
- **High face detection accuracy** (470+ detections in testing)
- **Real-time processing** (30 FPS)
- **Robust fallback options** (multiple YOLOv5 models)
- **Enhanced preprocessing** (brightness/contrast adjustment)

## 🎯 How It Works

1. **Face Detection**: YOLOv5 detects faces in real-time
2. **Pose Estimation**: CNN model predicts head orientation angles
3. **Rotation Logic**: Determines optimal screen rotation based on thresholds
4. **Manual Guidance**: Provides step-by-step rotation instructions
5. **Visual Feedback**: Shows real-time pose angles and recommendations

## 🔄 Rotation Logic

- **Roll > 15°**: Rotate to 270° (portrait flipped)
- **Roll < -15°**: Rotate to 90° (portrait)
- **Pitch > 12°**: Rotate to 180° (landscape flipped)
- **Pitch < -12°**: Rotate to 0° (landscape)
- **Default**: 0° (landscape)

## 📝 Notes

- **Manual Rotation Required**: Automatic screen rotation may not work on all systems
- **Windows Only**: Screen rotation functionality is Windows-specific
- **Webcam Required**: System needs access to webcam for face detection
- **Lighting Important**: Good lighting improves face detection accuracy

## 🤝 Support

For issues and troubleshooting:
1. Check `SCREEN_ROTATION_TROUBLESHOOTING.md`
2. Review `SOLUTION_SUMMARY.md`
3. Ensure all dependencies are installed
4. Verify Windows compatibility

## 📄 License

This project is for educational and research purposes.