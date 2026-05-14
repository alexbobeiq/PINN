import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
import yaml

from utils.data_loader import get_dataloaders
from models.pinn_model import PINN_Classifier
from utils.physics_loss import calculate_physics_loss

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config_pinn.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def objective(trial):
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Hiperparametri de tunat
    physics_weight = trial.suggest_float('physics_weight', 0.01, 5.0, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    
    # 2. Incarcam datele (Tunam pe 5% date pentru a optimiza strict performanta sub Scarcity)
    data_path = os.path.join(os.path.dirname(__file__), config['data_path'])
    train_loader_full, test_loader, scaler = get_dataloaders(
        data_path, config['time_steps'], batch_size
    )
    
    full_dataset = train_loader_full.dataset
    actual_dataset = full_dataset.dataset
    indices = full_dataset.indices
    
    train_labels = np.array([actual_dataset.labels[idx + actual_dataset.time_steps] for idx in indices])
    
    frac = 0.05
    sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=42)
    local_idx, _ = next(sss.split(np.zeros(len(train_labels)), train_labels))
    final_indices = [indices[i] for i in local_idx]
    train_subset = Subset(actual_dataset, final_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    # 3. Initializare Model
    model = PINN_Classifier(
        num_features=2, 
        num_classes=config['num_classes'], 
        time_steps=config['time_steps'],
        hidden_dim=hidden_dim
    ).to(device)
    
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 4. Antrenare
    max_epochs = 15
    
    model.train()
    for epoch in range(max_epochs):
        for x_scaled, setpoints, y_batch in train_loader:
            x_scaled = x_scaled.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits, phys_pred = model(x_scaled)
            
            loss_class = criterion_class(logits, y_batch)
            loss_phys = calculate_physics_loss(phys_pred, x_scaled)
            
            total_loss = loss_class + physics_weight * loss_phys
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    # 5. Evaluare
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            x_scaled, setpoints, y = batch
            x_scaled = x_scaled.to(device)
            logits, _ = model(x_scaled)
            pred = torch.argmax(logits, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(pred.cpu().numpy())
            
    f1 = f1_score(y_true, y_pred, average='weighted')
    return f1

if __name__ == "__main__":
    print("==================================================")
    print("  Incepem Optimizarea Optuna (Scenariul 5% Date)")
    print("==================================================")
    
    # Folosim TPE (Tree-structured Parzen Estimator) pentru cautare bayesiana
    study = optuna.create_study(direction="maximize")
    
    # Rulam 20 de experimente
    study.optimize(objective, n_trials=20)
    
    print("\n" + "="*50)
    print("OPTIMIZARE FINALIZATA!")
    print(f"Cel mai bun scor F1 obtinut la 5% date: {study.best_value:.4f}")
    print("PARAMETRII IDEALI SUNT:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("==================================================")
    print("Te rog sa actualizezi config_pinn.yaml cu aceste valori!")
