import torch
import torch.nn as nn
import torch.optim as optim
from models.data_driven import CNN1D_FaultClassifier

def train_cnn(train_loader, config, device):
    model = CNN1D_FaultClassifier(
        num_features=2,
        num_classes=config['num_classes'],
        time_steps=config['time_steps'],
        backbone_channels=config.get('backbone_channels', [16, 32, 64])
    ).to(device)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['learning_rate'] / 20
    )

    if device.type == 'cuda' and hasattr(torch, 'compile'):
        model = torch.compile(model)

    use_amp    = device.type == 'cuda'
    amp_scaler = torch.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for x_scaled, _, y_batch in train_loader:
            x_scaled = x_scaled.to(device, non_blocking=True)
            y_batch  = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x_scaled)
                loss   = criterion(logits, y_batch)

            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

            running_loss += loss.item()

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f"    [CNN] Epoch {epoch+1}/{config['epochs']} Loss: {running_loss/len(train_loader):.4f}  lr={lr_now:.2e}")

    return model
