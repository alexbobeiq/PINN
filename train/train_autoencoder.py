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

    epochs    = config.get('ae_epochs', 50)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=config['learning_rate'] / 20
    )

    use_amp    = device.type == 'cuda'
    amp_scaler = torch.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for x_scaled, _, _ in train_loader:
            x_scaled = x_scaled.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = nn.functional.mse_loss(model(x_scaled), x_scaled)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            total += loss.item()
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f"    [AE] Epoch {epoch+1}/{epochs}  Loss: {total/len(train_loader):.5f}  lr={lr_now:.2e}")

    return model
