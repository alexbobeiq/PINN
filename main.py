import yaml
from utils.data_loader import get_dataloaders
from train.train_data_driven import train_cnn
from train.train_pinn import train_pinn

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main():
    print("1. Încărcare configurări...")
    config_cnn = load_config('configs/config_cnn.yaml')
    config_pinn = load_config('configs/config_pinn.yaml')
    
    print("2. Procesare date (Inclusiv corecție underflow)...")
    train_loader, test_loader, scaler = get_dataloaders(
        data_path=config_cnn['data_path'], 
        time_steps=config_cnn['time_steps'], 
        batch_size=config_cnn['batch_size']
    )
    
    print("\n=== START Antrenament CNN (Data-Driven) ===")
    cnn_model = train_cnn(train_loader, config_cnn)
    
    print("\n=== START Antrenament PINN (Constrângeri Fizice) ===")
    pinn_model = train_pinn(train_loader, config_pinn)
    
    print("\nAntrenamentul s-a încheiat cu succes pentru ambele modele!")
    # Următorul pas: evaluezi cnn_model și pinn_model în evaluate/compare_models.py

if __name__ == "__main__":
    main()
