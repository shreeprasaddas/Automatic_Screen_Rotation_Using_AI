import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class BIWIDataset(Dataset):
    def __init__(self, dataset_path):
        self.samples = []

        for subject_folder in os.listdir(dataset_path):
            subject_path = os.path.join(dataset_path, subject_folder)
            if not os.path.isdir(subject_path):
                continue

            frame_files = sorted([f for f in os.listdir(subject_path) if f.endswith('_rgb.png')])
            print(f"🔍 Found {len(frame_files)} frames in {subject_path}")

            for frame_file in frame_files:
                pose_file = frame_file.replace("_rgb.png", "_pose.txt")
                frame_path = os.path.join(subject_path, frame_file)
                pose_path = os.path.join(subject_path, pose_file)

                if os.path.isfile(pose_path):
                    self.samples.append((frame_path, pose_path))
                else:
                    print(f"❌ Missing pose file for: {frame_file}")

        print(f"✅ Total image-pose pairs: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pose_path = self.samples[idx]

        # Load image and convert to tensor
        image = Image.open(img_path).convert("RGB").resize((128, 128))
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Load pose values
        with open(pose_path, "r") as f:
            pose = f.readline().strip().split()
            pose = torch.tensor([float(val) for val in pose[:3]], dtype=torch.float32)

        return image, pose
