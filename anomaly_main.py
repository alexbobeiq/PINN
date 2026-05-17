import yaml
import torch
import os
from utils.data_loader import get_dataloaders
from evaluate.anomaly_detection import run_anomaly_experiment


def load_config(path):
    base = os.path.dirname(__file__)
    with open(os.path.join(base, path)) as f:
        return yaml.safe_load(f)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    config = load_config('configs/config_anomaly.yaml')

    train_loader, test_loader, _ = get_dataloaders(
        os.path.join(os.path.dirname(__file__), config['data_path']),
        config['time_steps'],
        config['batch_size']
    )

    run_anomaly_experiment(
        full_train_subset=train_loader.dataset,
        test_loader=test_loader,
        config=config,
        device=device
    )


if __name__ == '__main__':
    main()
