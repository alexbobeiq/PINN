import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D_Autoencoder(nn.Module):
    def __init__(self, num_features=2, time_steps=60, backbone_channels=None):
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [32, 64, 128]
        ch1, ch2, ch3 = backbone_channels
        self.time_steps = time_steps

        self.enc1 = nn.Sequential(nn.Conv1d(num_features, ch1, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.enc2 = nn.Sequential(nn.Conv1d(ch1, ch2, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.enc3 = nn.Sequential(nn.Conv1d(ch2, ch3, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))

        self.dec1 = nn.Sequential(nn.ConvTranspose1d(ch3, ch2, 2, stride=2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose1d(ch2, ch1, 2, stride=2), nn.ReLU())
        self.dec3 = nn.ConvTranspose1d(ch1, num_features, 2, stride=2)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.enc1(h)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.dec1(h)
        h = self.dec2(h)
        h = self.dec3(h)
        # Interpolam inapoi la dimensiunea originala (60 nu e multiplu de 8)
        h = F.interpolate(h, size=self.time_steps, mode='linear', align_corners=False)
        return h.transpose(1, 2)

    def anomaly_score(self, x):
        with torch.no_grad():
            x_hat = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=[1, 2])