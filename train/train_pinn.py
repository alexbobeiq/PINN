import torch
import torch.nn as nn
import torch.optim as optim
from models.pinn_model import PINN_Classifier
from utils.physics_loss import calculate_physics_loss

def train_pinn(train_loader, config, device):
    # Utilizăm device-ul pasat din main
    model = PINN_Classifier(input_dim=2, num_classes=config['num_classes']).to(device)
    
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    model.train()
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits, phys_pred = model(x_batch)
            
            loss_class = criterion_class(logits, y_batch)
            loss_phys = calculate_physics_loss(phys_pred, x_batch)
            
            total_loss = loss_class + config['physics_weight'] * loss_phys
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            
    return model