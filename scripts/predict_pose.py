import torch
import cv2
import numpy as np
from scripts.train_pose_model import PoseNet


model = PoseNet()
model.load_state_dict(torch.load("models/head_pose_cnn.pth", map_location=torch.device('cpu')))
model.eval()

def predict_pose(face_img):
    img = cv2.resize(face_img, (128, 128))
    img = img.transpose(2, 0, 1) / 255.0
    input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor).squeeze().numpy()
    yaw, pitch, roll = output
    return yaw, pitch, roll