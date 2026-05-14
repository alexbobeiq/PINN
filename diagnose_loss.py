"""
Script de diagnosticare: masoara magnitudinea loss-urilor 
pentru a calibra physics_weight corect.
"""
import torch
import torch.nn as nn
import yaml, os
from utils.data_loader import get_dataloaders
from models.pinn_model import PINN_Classifier
from utils.physics_loss import calculate_physics_loss

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'configs/config_pinn.yaml')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader, scaler = get_dataloaders(
    os.path.join(os.path.dirname(__file__), config['data_path']),
    config['time_steps'], config['batch_size']
)

model = PINN_Classifier(num_features=2, num_classes=config['num_classes'],
                        time_steps=config['time_steps'], hidden_dim=config.get('hidden_dim', 256)).to(device)
criterion = nn.CrossEntropyLoss()

# Masuram loss-urile pe primele 5 batch-uri (model NEANTRENANT)
model.train()
for i, (x, sp, y) in enumerate(train_loader):
    if i >= 5: break
    x, y = x.to(device), y.to(device)
    logits, phys_params = model(x)
    loss_class = criterion(logits, y)
    loss_phys = calculate_physics_loss(phys_params, x)
    print(f"Batch {i}: CrossEntropy={loss_class.item():.4f}  PhysicsLoss={loss_phys.item():.4f}  "
          f"Ratio CE/Phys={loss_class.item()/max(loss_phys.item(), 1e-8):.2f}")

print(f"\n=> Daca Ratio >> 1, physics_weight trebuie CRESCUT")
print(f"=> Daca Ratio << 1, physics_weight trebuie SCAZUT")
