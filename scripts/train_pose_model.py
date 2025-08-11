import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import BIWIDataset

# CNN Model
class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),   # 128 → 63
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),  # 63 → 30
            nn.Flatten(),
            nn.Linear(32 * 30 * 30, 128), nn.ReLU(),
            nn.Linear(128, 3)  # 3D pose: yaw, pitch, roll
        )

    def forward(self, x):
        return self.net(x)

def main():
    dataset_path = "dataset/BIWI/faces_0"
    print("📦 Loading dataset...")
    dataset = BIWIDataset(dataset_path)
    dataset.samples = dataset.samples[:2000]  # You can also try 500, 1000, etc.

    if len(dataset) == 0:
        raise ValueError("❌ No image-pose samples found. Check dataset format!")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ✅ Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    model = PoseNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ Training Loop
    for epoch in range(20):
        total_loss = 0.0
        model.train()

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📚 Epoch [{epoch + 1}/20] | 🔻 Loss: {total_loss:.4f}")

    # ✅ Save Model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/head_pose_cnn.pth")
    print("✅ Model saved as models/head_pose_cnn.pth")

if __name__ == "__main__":
    main()
