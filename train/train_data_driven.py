import torch
import torch.nn as nn
import torch.optim as optim
from models.data_driven import CNN1D_FaultClassifier

def train_cnn(train_loader, config, device):
    # Nu mai detectăm device-ul aici, îl primim din main.py
    model = CNN1D_FaultClassifier(num_features=2, num_classes=config['num_classes']).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    model.train()
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            # Mutăm datele pe dispozitivul corect
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    return model