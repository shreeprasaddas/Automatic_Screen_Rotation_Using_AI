#!/usr/bin/env python3
"""
Simple Linux Screen Rotation System with Trained Model
Uses the trained rotation model for screen rotation prediction and applies it on Linux.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import time
import os
import subprocess
import threading
from collections import deque

class RotationModel(nn.Module):
    """CNN model for screen rotation prediction"""
    
    def __init__(self, num_classes=4):
        super(RotationModel, self).__init__()

        # Use pre-trained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=False)
        
        # Modify the final layer for our number of classes
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_rotation_model():
    """Load the trained rotation model"""
    try:
        # Configure torch for smoother CPU performance in VMs
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        model = RotationModel(num_classes=4).to(device)
        
        # Load checkpoint
        checkpoint = torch.load('models/best_rotation_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Rotation prediction model loaded")
        print(f"📊 Model accuracy: {checkpoint.get('val_acc', 'N/A')}")
        print(f"🖥️  Using device: {device}")
        
        return model, device
    except Exception as e:
        print(f"❌ Failed to load rotation model: {e}")
        return None, None

def configure_camera(cap: cv2.VideoCapture):
    """Apply performance-friendly camera settings."""
    try:
        # Prefer moderate resolution for speed in VM environments
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Lower FPS to reduce CPU load; VM webcams often benefit from 15-30 FPS
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Use MJPG to reduce CPU decoding overhead (if supported by device)
        try:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass
        # Reduce internal buffer to get the freshest frame (if supported)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    except Exception as e:
        print(f"⚠️ Camera configuration warning: {e}")

class ThreadedCamera:
    """Continuously capture frames on a background thread to reduce UI lag."""
    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        self.is_opened = self.cap.isOpened()
        self.lock = threading.Lock()
        self.latest = None
        self.stopped = False
        if self.is_opened:
            configure_camera(self.cap)
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

    def _reader(self):
        while not self.stopped and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest = frame

    def read(self):
        with self.lock:
            frame = None if self.latest is None else self.latest.copy()
        return frame is not None, frame

    def release(self):
        self.stopped = True
        try:
            if hasattr(self, 'thread'):
                self.thread.join(timeout=0.5)
        except Exception:
            pass
        if self.cap:
            self.cap.release()

def get_primary_display():
    """Get the primary display name for Linux"""
    try:
        result = subprocess.run(['xrandr', '--current'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        for line in lines:
            # Look for connected displays, prioritize primary
            if 'connected' in line and 'disconnected' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    display = parts[0]
                    if 'primary' in line:
                        print(f"✅ Found primary display: {display}")
                        return display
                    elif 'Virtual1' in line:  # Fallback for VM
                        print(f"✅ Found VM display: {display}")
                        return display
        
        # If no primary found, try any connected display
        for line in lines:
            if 'connected' in line and 'disconnected' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    display = parts[0]
                    print(f"✅ Found connected display: {display}")
                    return display
        
        print("❌ No display found")
        return None
    except Exception as e:
        print(f"❌ Error getting display: {e}")
        return None

def rotate_screen_linux(rotation_angle):
    """Rotate screen on Linux using xrandr"""
    try:
        display = get_primary_display()
        if not display:
            print("❌ No display available for rotation")
            return False
        
        # Map rotation angle to xrandr rotation value
        rotation_map = {
            0: 'normal',
            90: 'left',
            180: 'inverted',
            270: 'right'
        }
        
        rotation_value = rotation_map.get(rotation_angle, 'normal')
        
        # Execute xrandr command
        cmd = ['xrandr', '--output', display, '--rotate', rotation_value]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Screen rotated to {rotation_angle}° ({rotation_value})")
            return True
        else:
            print(f"❌ Failed to rotate screen: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error rotating screen: {e}")
        return False

def preprocess_for_rotation(image, target_size=(224, 224)):
    """Preprocess image for rotation model"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        image = Image.fromarray(image)
    
    return transform(image).unsqueeze(0)

def predict_rotation_from_image(rotation_model, image, device):
    """Predict rotation from image"""
    if rotation_model is None:
        return None, 0.0, None
    
    with torch.no_grad():
        image_tensor = preprocess_for_rotation(image).to(device)
        outputs = rotation_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Map class to rotation
    rotation_map = {
        0: "0° (Normal)",
        1: "90° (Right)",
        2: "180° (Down)",
        3: "270° (Left)"
    }
    
    return rotation_map[predicted_class], confidence, probabilities[0].cpu().numpy()

def show_rotation_instructions():
    """Show manual rotation instructions for Linux"""
    print("\n" + "="*60)
    print("🔄 MANUAL SCREEN ROTATION INSTRUCTIONS FOR LINUX")
    print("="*60)
    print("If automatic rotation doesn't work, here are manual methods:")
    print()
    print("⌨️  Method 1: xrandr Command Line")
    print("   1. Open terminal")
    print("   2. Run: xrandr --output <display> --rotate <direction>")
    print("   3. Directions: normal, right, inverted, left")
    print()
    print("🎮 Method 2: Display Settings")
    print("   1. Go to Settings → Displays")
    print("   2. Look for 'Rotation' or 'Orientation'")
    print("   3. Select your desired rotation")
    print()
    print("💻 Method 3: Graphics Driver Control Panel")
    print("   1. Right-click on desktop → Graphics options")
    print("   2. Look for 'Rotation' settings")
    print("   3. Select your desired rotation")
    print()
    print("💡 Tips:")
    print("   - Some systems may require sudo privileges")
    print("   - Graphics drivers must support rotation")
    print("   - External monitors may have different rotation options")
    print("   - If rotation doesn't work, try updating graphics drivers")
    print("="*60)

def main():
    print("🎯 Simple Linux Screen Rotation System with Trained Model")
    print("=" * 60)
    
    # Load rotation model
    rotation_model, device = load_rotation_model()
    
    if rotation_model is None:
        print("❌ No rotation model available")
        return
    
    # Test display detection
    print("🔍 Testing display detection...")
    display = get_primary_display()
    if not display:
        print("❌ Could not detect primary display")
        return
    print(f"✅ Display detection successful: {display}")
    
    # Initialize camera (threaded)
    print("📹 Initializing camera...")
    cam = ThreadedCamera(0)
    if not cam.is_opened:
        print("📹 Trying alternative camera indices...")
        for i in range(1, 5):
            cam = ThreadedCamera(i)
            if cam.is_opened:
                print(f"✅ Camera opened at index {i}")
                break

    if not cam.is_opened:
        print("❌ Could not open any camera")
        print("💡 Make sure your camera is connected and not being used by another application")
        return
    print("✅ Camera initialized successfully")
    
    print("\n🎮 Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to show rotation instructions")
    print("  - Press 'p' to show/hide probabilities")
    print("  - Press 'c' to show/hide confidence")
    print("  - Press 't' to test manual rotation")
    
    show_probabilities = True
    show_confidence = True
    
    # Statistics
    frame_count = 0
    rotation_predictions = 0
    last_rotation = None
    
    print("\n🚀 Starting real-time rotation prediction...")
    print("📹 Point your face at the camera to get rotation predictions")
    
    # Inference throttling and smoothing
    infer_every_n = 3  # run inference every N frames
    recent_angles: deque[int] = deque(maxlen=5)

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                print("❌ Failed to capture frame")
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # Predict rotation intermittently to reduce CPU usage
            rotation = None
            confidence = 0.0
            probabilities = None
            if frame_count % infer_every_n == 0:
                rotation, confidence, probabilities = predict_rotation_from_image(rotation_model, frame, device)
            
            if rotation:
                rotation_predictions += 1
                
                # Extract rotation angle for screen rotation
                rotation_angle = int(rotation.split('°')[0])

                # Smoothing: use most common recent angle before applying
                recent_angles.append(rotation_angle)
                smoothed_angle = max(set(recent_angles), key=recent_angles.count)
                
                # Apply screen rotation if it changed
                if smoothed_angle != last_rotation:
                    print(f"🔄 Applying rotation: {smoothed_angle}° (smoothed from {rotation_angle}°)")
                    if rotate_screen_linux(smoothed_angle):
                        last_rotation = smoothed_angle
                    else:
                        print("⚠️  Screen rotation failed, continuing...")
                
                # Display main prediction
                cv2.putText(frame, f"ROTATION: {rotation}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if show_confidence:
                    cv2.putText(frame, f"Confidence: {confidence:.3f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display probabilities
                if show_probabilities and probabilities is not None:
                    rotations = ["0°", "90°", "180°", "270°"]
                    for i, (rot, prob) in enumerate(zip(rotations, probabilities)):
                        y_pos = 90 + i * 25
                        color = (0, 255, 0) if i == rotations.index(rotation.split()[0]) else (255, 255, 255)
                        cv2.putText(frame, f"{rot}: {prob:.3f}", (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Display statistics
            cv2.putText(frame, f"Frames: {frame_count}", (10, frame.shape[0]-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Predictions: {rotation_predictions}", (10, frame.shape[0]-45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display controls info
            cv2.putText(frame, "Press 'q' to quit, 'r' for rotation help", (10, frame.shape[0]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            cv2.imshow('Simple Linux Screen Rotation System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                show_rotation_instructions()
            elif key == ord('p'):
                show_probabilities = not show_probabilities
                print(f"Probabilities: {'ON' if show_probabilities else 'OFF'}")
            elif key == ord('c'):
                show_confidence = not show_confidence
                print(f"Confidence: {'ON' if show_confidence else 'OFF'}")
            elif key == ord('t'):
                print("🧪 Testing manual rotation...")
                test_rotations = [0, 90, 180, 270]
                for angle in test_rotations:
                    print(f"Testing {angle}° rotation...")
                    if rotate_screen_linux(angle):
                        time.sleep(1)  # Wait a bit to see the rotation
                    else:
                        print(f"Failed to rotate to {angle}°")
                print("Manual rotation test complete")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        cam.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        print(f"\n📊 Session Summary:")
        print(f"   Model used: trained rotation model")
        print(f"   Display: {display}")
        print(f"   Total frames: {frame_count}")
        print(f"   Rotation predictions: {rotation_predictions}")
        if frame_count > 0:
            print(f"   Prediction rate: {100 * rotation_predictions / frame_count:.1f}%")
        print("✅ System shutdown complete")

if __name__ == "__main__":
    main() 