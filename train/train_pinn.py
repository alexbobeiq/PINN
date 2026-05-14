import torch
import torch.nn as nn
import torch.optim as optim
from models.pinn_model import PINN_Classifier
from utils.physics_loss import calculate_physics_loss

def train_pinn(train_loader, config, device):
    model = PINN_Classifier(
        num_features=2,
        num_classes=config['num_classes'],
        time_steps=config['time_steps'],
        backbone_channels=config.get('backbone_channels', [16, 32, 64])
    ).to(device)

    criterion         = nn.CrossEntropyLoss()
    optimizer         = optim.Adam(model.parameters(), lr=config['learning_rate'])
    physics_weight_max = config.get('physics_weight', 0.1)
    warmup_epochs      = config.get('physics_warmup_epochs', config['epochs'] // 2)

    model.train()
    for epoch in range(config['epochs']):
        # Linear ramp: 0 → physics_weight_max over warmup_epochs
        t = min(epoch / max(warmup_epochs, 1), 1.0)
        physics_weight = t * physics_weight_max

        running_loss = 0.0
        for x_scaled, _, y_batch in train_loader:
            x_scaled = x_scaled.to(device)
            y_batch  = y_batch.to(device)

            optimizer.zero_grad()
            logits, phys_params = model(x_scaled)

            loss_class = criterion(logits, y_batch)
            loss_phys  = calculate_physics_loss(phys_params, x_scaled)
            loss       = loss_class + physics_weight * loss_phys

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        print(f"    [PINN] Epoch {epoch+1}/{config['epochs']} "
              f"phys_w={physics_weight:.3f}  Loss: {running_loss/len(train_loader):.4f}")

    return model
