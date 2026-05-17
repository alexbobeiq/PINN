import yaml
import torch
import os
from utils.data_loader import get_dataloaders
from utils.checkpoints import load_models, save_models
from evaluate.noise_robustness import run_noise_experiment
from train.train_data_driven import train_cnn
from train.train_pinn import train_pinn


def load_config(path):
    base = os.path.dirname(__file__)
    with open(os.path.join(base, path)) as f:
        return yaml.safe_load(f)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    config_cnn  = load_config('configs/config_cnn.yaml')
    config_pinn = load_config('configs/config_pinn.yaml')

    train_loader, test_loader, scaler = get_dataloaders(
        os.path.join(os.path.dirname(__file__), config_cnn['data_path']),
        config_cnn['time_steps'],
        config_cnn['batch_size']
    )

    # Incarca modele salvate daca exista, altfel antreneaza si salveaza
    loaded = load_models(device)
    if loaded:
        cnn_model, pinn_model, _ = loaded
        print("[noise_main] Modele incarcate din checkpoint — skip antrenament.\n")
    else:
        print("[noise_main] Checkpoint negasit — antrenez de la zero...\n")
        cnn_model  = train_cnn(train_loader, config_cnn, device)
        pinn_model = train_pinn(train_loader, config_pinn, device)
        save_models(cnn_model, pinn_model, scaler, config_cnn)

    # Curba completa de robustete la zgomot:
    #   - Test la sigma crescator: 0 → 2.0
    #   - Crossover-ul (sigma dupa care PINN bate CNN) e argumentul principal
    run_noise_experiment(
        train_loader=train_loader,
        test_loader=test_loader,
        config_cnn=config_cnn,
        config_pinn=config_pinn,
        device=device,
        noise_levels=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        n_seeds=5,
        train_sigma=0.0,
        train_fraction=1.0,
        pretrained_cnn=cnn_model,
        pretrained_pinn=pinn_model,
    )


if __name__ == '__main__':
    main()
