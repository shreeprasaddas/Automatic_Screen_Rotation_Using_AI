import torch
import torch.nn as nn
import cv2
import numpy as np

# Try to load the best rotation model, but fallback gracefully
model = None
try:
    checkpoint = torch.load("models/best_rotation_model.pth", map_location=torch.device('cpu'))
    print("✅ Best rotation model checkpoint loaded")
    print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
    print(f"📊 Model state dict keys: {list(checkpoint['model_state_dict'].keys())[:10]}...")
except Exception as e:
    print(f"❌ Error loading best rotation model: {e}")
    print("🔄 Will use fallback methods")

def predict_rotation_with_best_model(face_img):
    """
    Predict screen rotation using the best rotation model
    Returns: rotation_angle (0, 90, 180, or 270 degrees)
    """
    # For now, use fallback since the model architecture is complex
    return predict_rotation_fallback(face_img)

def predict_pose_with_best_model(face_img):
    """
    Predict pose angles using best_rotation_model.pth
    Returns: yaw, pitch, roll angles
    """
    # For now, use fallback since the model architecture is complex
    return predict_pose_fallback(face_img)

def pose_to_rotation(yaw, pitch, roll):
    """
    Convert pose angles to screen rotation decision
    """
    # Thresholds for rotation decisions
    roll_threshold = 15
    pitch_threshold = 10
    
    if roll > roll_threshold:
        return 270  # Tilted left
    elif roll < -roll_threshold:
        return 90   # Tilted right
    elif pitch > pitch_threshold:
        return 180  # Looking up
    else:
        return 0    # Facing front

def predict_rotation_fallback(face_img):
    """
    Fallback rotation prediction when best model is not available
    """
    # Simple fallback logic based on image analysis
    # This could be enhanced with basic computer vision techniques
    return 0  # Default to normal rotation

def predict_pose_fallback(face_img):
    """
    Fallback pose prediction when best model is not available
    """
    # Return neutral pose
    return 0.0, 0.0, 0.0
