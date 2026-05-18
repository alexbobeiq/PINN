import yaml
import torch
import os
from utils.data_loader import get_dataloaders
from evaluate.anomaly_detection import run_anomaly_noise_experiment

BASE = os.path.dirname(__file__)


def load_config(rel_path):
    with open(os.path.join(BASE, rel_path)) as f:
        return yaml.safe_load(f)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    config = load_config('configs/config_anomaly.yaml')

    train_loader, test_loader, _ = get_dataloaders(
        os.path.join(BASE, config['data_path']),
        config['time_steps'],
        config['batch_size']
    )

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    print("=" * 65)
    print("NOISE ROBUSTNESS — PINN vs Autoencoder")
    print("Ambele modele antrenate pe date Normale curate.")
    print("Evaluare AUROC la diferite niveluri de zgomot Gaussian.")
    print("=" * 65)

    results = run_anomaly_noise_experiment(
        full_train_subset=train_loader.dataset,
        test_loader=test_loader,
        config=config,
        device=device,
        noise_levels=noise_levels,
        n_seeds=3,
    )

    print("\n" + "=" * 65)
    print("REZULTATE FINALE:")
    print(f"{'Sigma':>8} | {'PINN AUROC':>12} | {'AE AUROC':>10} | Castigator")
    print("-" * 65)
    for i, sigma in enumerate(noise_levels):
        pm = results['pinn_auroc_mean'][i]
        am = results['ae_auroc_mean'][i]
        winner = "PINN ✓" if pm > am else "AE  ✓"
        print(f"{sigma:>8.2f} | {pm:>12.4f} | {am:>10.4f} | {winner}")


if __name__ == '__main__':
    main()
