import torch
import torch.nn as nn
import torch.optim as optim
from models.data_driven import CNN1D_FaultClassifier

def train_cnn(train_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # type: ignore
    model = CNN1D_FaultClassifier(time_steps=config['time_steps'], num_classes=config['num_classes']).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    model.train()
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"CNN Epoch [{epoch+1}/{config['epochs']}] | Loss: {running_loss/len(train_loader):.4f}")
    return model
