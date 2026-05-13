import torch.nn as nn

class PINN_Classifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_classes=8):
        super(PINN_Classifier, self).__init__()
        
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.class_head = nn.Linear(hidden_dim, num_classes)
        # Physics head fără activare finală pentru a putea atinge valoarea 2500
        self.physics_head = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        last_step_pressures = x[:, -1, 0:2] 
        features = self.shared_net(last_step_pressures)
        
        logits = self.class_head(features)
        phys_prediction = self.physics_head(features)
        return logits, phys_prediction