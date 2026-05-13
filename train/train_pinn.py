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
        hidden_dim=config.get('hidden_dim', 64)
    ).to(device)
    
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    physics_weight = config.get('physics_weight', 0.5)
    max_epochs = config['epochs']
    
    model.train()
    for epoch in range(max_epochs):
        running_loss = 0.0
        
        for x_scaled, setpoints, y_batch in train_loader:
            x_scaled = x_scaled.to(device)
            setpoints = setpoints.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits, phys_pred = model(x_scaled)
            
            loss_class = criterion_class(logits, y_batch)
            loss_phys = calculate_physics_loss(phys_pred, x_scaled)
            
            total_loss = loss_class + physics_weight * loss_phys
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += total_loss.item()
            
        print(f"    [PINN] Epoch {epoch+1}/{max_epochs} Loss: {running_loss/len(train_loader):.4f}")
        
    return model