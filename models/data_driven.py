import torch
import torch.nn as nn

class CNN1D_FaultClassifier(nn.Module):
    def __init__(self, num_features=2, num_classes=8, time_steps=60):
        super(CNN1D_FaultClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.flatten = nn.Flatten()
        linear_input = 64 * (time_steps // 4) 
        self.fc1 = nn.Linear(linear_input, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [batch, time_steps, 2] — uses only the 2 pressure features
        x = x.transpose(1, 2)  # [batch, features, time_steps]
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)