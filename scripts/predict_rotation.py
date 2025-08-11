import torch
import cv2
import numpy as np
from scripts.train_pose_model import PoseNet

# Load the best rotation model
model = PoseNet()
checkpoint = torch.load("models/best_rotation_model.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def predict_rotation(face_img):
    """
    Predict screen rotation directly from face image using best_rotation_model.pth
    Returns: rotation_angle (0, 90, 180, or 270 degrees)
    """
    # Preprocess image
    img = cv2.resize(face_img, (128, 128))
    img = img.transpose(2, 0, 1) / 255.0
    input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor).squeeze().numpy()
    
    # The model outputs pose angles, we need to convert to rotation
    yaw, pitch, roll = output
    
    # Convert pose angles to rotation decision
    rotation = pose_to_rotation(yaw, pitch, roll)
    
    return rotation

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

def predict_pose_with_best_model(face_img):
    """
    Predict pose angles using best_rotation_model.pth
    Returns: yaw, pitch, roll angles
    """
    # Preprocess image
    img = cv2.resize(face_img, (128, 128))
    img = img.transpose(2, 0, 1) / 255.0
    input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor).squeeze().numpy()
    
    yaw, pitch, roll = output
    return yaw, pitch, roll
