import torch
import torch.optim as optim
from models.pinn_anomaly import PINN_AnomalyDetector
from utils.physics_loss import calculate_physics_loss


def train_pinn_anomaly(train_loader, config, device):
    model = PINN_AnomalyDetector(
        num_features=2,
        time_steps=config['time_steps'],
        backbone_channels=config.get('backbone_channels', [32, 64, 128])
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    epochs    = config.get('pinn_anomaly_epochs', 50)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for x_scaled, _, _ in train_loader:
            x_scaled = x_scaled.to(device)
            optimizer.zero_grad()
            phys_params = model(x_scaled)
            loss = calculate_physics_loss(phys_params, x_scaled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        print(f"    [PINN-AD] Epoch {epoch+1}/{epochs}  Loss: {total/len(train_loader):.5f}")

    return model
