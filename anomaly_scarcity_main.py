import yaml
import torch
import os
from utils.data_loader import get_dataloaders
from evaluate.anomaly_detection import run_anomaly_scarcity_experiment

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

    fractions = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]

    print("=" * 65)
    print("SCARCITY ANOMALY DETECTION — PINN vs Autoencoder")
    print("Ambele modele antrenate EXCLUSIV pe date Normale.")
    print("Argumentul: PINN mentine AUROC cu putine date datorita fizicii.")
    print("=" * 65)

    results = run_anomaly_scarcity_experiment(
        full_train_subset=train_loader.dataset,
        test_loader=test_loader,
        config=config,
        device=device,
        fractions=fractions,
        n_seeds=5,
    )

    print("\n" + "=" * 65)
    print("REZULTATE FINALE:")
    print(f"{'Fractie':>8} | {'N Normal':>8} | {'PINN AUROC':>12} | {'AE AUROC':>10} | Castigator")
    print("-" * 65)
    n_total = len(train_loader.dataset)
    for i, frac in enumerate(fractions):
        n = max(32, int(n_total * frac))
        pm = results['pinn_auroc_mean'][i]
        am = results['ae_auroc_mean'][i]
        winner = "PINN ✓" if pm > am else "AE  ✓"
        print(f"{frac:>8.3f} | {n:>8d} | {pm:>12.4f} | {am:>10.4f} | {winner}")


if __name__ == '__main__':
    main()
