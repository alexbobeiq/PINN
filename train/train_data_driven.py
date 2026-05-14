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
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    model.train()
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for batch in train_loader:
            # Datasetul returneaza (x_scaled, setpoints, y) — CNN ignora setpoints
            x_scaled, _, y_batch = batch
            x_scaled, y_batch = x_scaled.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_scaled)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"    [CNN] Epoch {epoch+1}/{config['epochs']} Loss: {running_loss/len(train_loader):.4f}")
        
    return model