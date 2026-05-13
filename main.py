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
        raise FileNotFoundError(f"Configurația nu a fost găsită la: {abs_path}")
    with open(abs_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Fix pentru CUDA Error 101
    device = torch.device('cpu')
    if torch.cuda.is_available():
        try:
            _ = torch.cuda.get_device_count()
            device = torch.device('cuda')
        except:
            print("CUDA raportat dar inaccesibil. Utilizăm CPU.")

    print(f"Sistemul rulează pe: {device}")

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
    fractions = [1.0, 0.3, 0.1, 0.05]
    run_scarcity_experiment(full_train_dataset, test_loader, config_cnn, config_pinn, device, fractions)

if __name__ == "__main__":
    main()