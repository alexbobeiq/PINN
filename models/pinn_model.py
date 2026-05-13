import torch
import torch.nn as nn

class PINN_Classifier(nn.Module):
    def __init__(self, num_features=2, num_classes=8, time_steps=60, hidden_dim=64):
        super(PINN_Classifier, self).__init__()
        
        # Aceeasi arhitectura de baza ca la CNN (pentru comparatie corecta 1 la 1)
        # Primeste doar cele 2 presiuni (FARA setpoint-uri, deci FARA data leakage)
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.flatten = nn.Flatten()
        conv_out_dim = 64 * (time_steps // 4) 
        
        # Head-ul clasic de clasificare
        self.fc1 = nn.Linear(conv_out_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Head-ul Fizic (Physics Head)
        # Incearca sa reconstruiasca integral toata curba de presiune (starea fizica a sistemului).
        # Asta forteaza reteaua sa memoreze toata fizica procesului, prevenind overfitting-ul la date mici.
        self.physics_head = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, time_steps * num_features) # 60 * 2 = 120
        )

    def forward(self, x):
        # x: [batch, time_steps, 2]
        x = x.transpose(1, 2)  # format pentru Conv1D
        
        # Extragerea trasaturilor (Features)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        features = self.flatten(x)
        
        # Ramura de Clasificare
        logits = self.fc2(self.relu(self.fc1(features)))
        
        # Ramura de Fizica
        phys_prediction = self.physics_head(features)
        
        return logits, phys_prediction