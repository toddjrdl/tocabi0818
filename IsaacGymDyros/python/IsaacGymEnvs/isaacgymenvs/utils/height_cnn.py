# height_cnn.py
import torch.nn as nn
import torch.nn.functional as F

class HeightCNN(nn.Module):
    def __init__(self, in_ch=1, emb_dim=24, H=11, W=7):
        super().__init__()
        # conv → pool → conv → pool → fc 구조
        self.conv1 = nn.Conv2d(in_ch, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # H, W는 height-scan 그리드 크기
        flat_dim = 16 * (H // 4) * (W // 4)
        self.fc    = nn.Linear(flat_dim, emb_dim)

    def forward(self, x):
        # x: (B, 1, H, W)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # → (B, 8, H/2, W/2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # → (B, 16, H/4, W/4)
        x = x.view(x.size(0), -1)
        return self.fc(x)       # → (B, emb_dim)