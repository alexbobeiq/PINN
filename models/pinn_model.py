import torch.nn as nn

class PINN_Classifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_classes=8):
        super(PINN_Classifier, self).__init__()
        # Folosim Tanh pentru a putea calcula derivate continue prin autograd
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # Capăt pentru Clasificarea Defectului
        self.class_head = nn.Linear(hidden_dim, num_classes)
        # Capăt pentru Estimarea Fizică (ex: prezice debitul sau forța)
        self.physics_head = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        # Ne concentrăm pe ultimul eșantion din fereastra de timp
        last_step_features = x[:, -1, :] 
        features = self.shared_net(last_step_features)
        
        logits = self.class_head(features)
        phys_prediction = self.physics_head(features)
        return logits, phys_prediction
