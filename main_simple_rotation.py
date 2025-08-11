#!/usr/bin/env python3
"""
Simple Screen Rotation System with Trained Model
Uses the trained rotation model for screen rotation prediction.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import time
import os

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
    """Show manual rotation instructions"""
    print("\n" + "="*60)
    print("🔄 MANUAL SCREEN ROTATION INSTRUCTIONS")
    print("="*60)
    print("Since automatic rotation may not work on all systems,")
    print("here are manual methods to rotate your screen:")
    print()
    print("📱 Method 1: Windows Display Settings")
    print("   1. Right-click on desktop → Display settings")
    print("   2. Scroll down to 'Display orientation'")
    print("   3. Select: Landscape, Portrait, Landscape (flipped), or Portrait (flipped)")
    print("   4. Click 'Keep changes' when prompted")
    print()
    print("⌨️  Method 2: Keyboard Shortcuts (if supported)")
    print("   - Ctrl + Alt + → : Rotate 90° right")
    print("   - Ctrl + Alt + ← : Rotate 90° left")
    print("   - Ctrl + Alt + ↓ : Rotate 180°")
    print("   - Ctrl + Alt + ↑ : Return to normal")
    print()
    print("🎮 Method 3: Graphics Driver Control Panel")
    print("   1. Right-click on desktop → Graphics options")
    print("   2. Look for 'Rotation' or 'Orientation' settings")
    print("   3. Select your desired rotation")
    print()
    print("💡 Tips:")
    print("   - Some systems may require administrator privileges")
    print("   - Graphics drivers must support rotation")
    print("   - External monitors may have different rotation options")
    print("   - If rotation doesn't work, try updating graphics drivers")
    print("="*60)

def main():
    print("🎯 Simple Screen Rotation System with Trained Model")
    print("=" * 60)
    
    # Load rotation model
    rotation_model, device = load_rotation_model()
    
    if rotation_model is None:
        print("❌ No rotation model available")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("\n🎮 Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to show rotation instructions")
    print("  - Press 'p' to show/hide probabilities")
    print("  - Press 'c' to show/hide confidence")
    
    show_probabilities = True
    show_confidence = True
    
    # Statistics
    frame_count = 0
    rotation_predictions = 0
    
    print("\n🚀 Starting real-time rotation prediction...")
    print("📹 Point your face at the camera to get rotation predictions")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            
            # Predict rotation from the entire frame
            rotation, confidence, probabilities = predict_rotation_from_image(rotation_model, frame, device)
            
            if rotation:
                rotation_predictions += 1
                
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
            
            cv2.imshow('Simple Screen Rotation System', frame)
            
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
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        print(f"\n📊 Session Summary:")
        print(f"   Model used: trained rotation model")
        print(f"   Total frames: {frame_count}")
        print(f"   Rotation predictions: {rotation_predictions}")
        if frame_count > 0:
            print(f"   Prediction rate: {100 * rotation_predictions / frame_count:.1f}%")
        print("✅ System shutdown complete")

if __name__ == "__main__":
    main() 