import torch
import torch.nn as nn
import torch.optim as optim
from models.autoencoder import CNN1D_Autoencoder


def train_autoencoder(train_loader, config, device):
    model = CNN1D_Autoencoder(
        num_features=2,
        time_steps=config['time_steps'],
        backbone_channels=config.get('backbone_channels', [32, 64, 128])
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    epochs    = config.get('ae_epochs', 50)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for x_scaled, _, _ in train_loader:
            x_scaled = x_scaled.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x_scaled), x_scaled)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"    [AE] Epoch {epoch+1}/{epochs}  Loss: {total/len(train_loader):.5f}")

    return model
