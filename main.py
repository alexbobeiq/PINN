import yaml
import torch
import os
from utils.data_loader import get_dataloaders
from train.train_data_driven import train_cnn
from train.train_pinn import train_pinn
from evaluate.compare_models import run_scarcity_experiment

def load_config(path):
    # Această linie transformă calea relativă într-o cale absolută față de main.py
    base_path = os.path.dirname(__file__)
    abs_path = os.path.join(base_path, path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config not found at: {abs_path}")
    with open(abs_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Sistemul ruleaza pe: {device}")

    # Încărcăm config-urile
    config_cnn = load_config('configs/config_cnn.yaml')
    config_pinn = load_config('configs/config_pinn.yaml')
    
    # Obținem datele
    train_loader, test_loader, scaler = get_dataloaders(
        os.path.join(os.path.dirname(__file__), config_cnn['data_path']), 
        config_cnn['time_steps'], 
        config_cnn['batch_size']
    )
    
    full_train_dataset = train_loader.dataset

    # Lansăm experimentul de scarcity
    fractions = [0.6, 0.4, 0.3, 0.1, 0.05]
    results = run_scarcity_experiment(
        full_train_dataset, test_loader, config_cnn, config_pinn, device, fractions
    )
    
    # Afișăm rezumatul final
    print("\n" + "="*60)
    print("REZUMAT FINAL")
    print("="*60)
    for i, frac in enumerate(fractions):
        cnn_f1 = results['cnn'][i]
        pinn_f1 = results['pinn'][i]
        winner = "PINN ✓" if pinn_f1 > cnn_f1 else "CNN ✓"
        print(f"  {frac*100:5.0f}% date: CNN={cnn_f1:.4f}  PINN={pinn_f1:.4f}  -> {winner}")

if __name__ == "__main__":
    main()