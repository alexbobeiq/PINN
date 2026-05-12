import torch
import torch.nn as nn
import torch.optim as optim
from models.pinn_model import PINN_Classifier
from utils.physics_loss import calculate_physics_loss

def train_pinn(train_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # type: ignore
    model = PINN_Classifier(num_classes=config['num_classes']).to(device)
    
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    phys_weight = config['physics_weight']
    
    model.train()
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Activăm urmărirea gradientului pentru variabilele de intrare (Esențial pt PINN)
            x_batch.requires_grad_(True) 
            
            optimizer.zero_grad()
            logits, phys_pred = model(x_batch)
            
            # Pierderea din date (clasificarea defectului)
            loss_class = criterion_class(logits, y_batch)
            
            # Pierderea din ecuațiile fizice
            loss_phys = calculate_physics_loss(phys_pred, x_batch)
            
            # Costul total (Data + Fizică)
            total_loss = loss_class + phys_weight * loss_phys
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        if (epoch+1) % 50 == 0:
            print(f"PINN Epoch [{epoch+1}/{config['epochs']}] | Total Loss: {running_loss/len(train_loader):.4f}")
    return model
