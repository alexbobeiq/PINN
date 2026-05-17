import yaml
import torch
import numpy as np
import os
from sklearn.metrics import f1_score

from utils.data_loader import get_dataloaders, get_external_test_loader
from utils.checkpoints import save_models
from train.train_data_driven import train_cnn
from train.train_pinn import train_pinn
from evaluate.compare_models import run_scarcity_experiment

BASE = os.path.dirname(__file__)


def load_config(rel_path):
    path = os.path.join(BASE, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config negasit: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_on_loader(model, loader, device, is_pinn):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, _, y in loader:
            x = x.to(device)
            logits = model(x)[0] if is_pinn else model(x)
            y_true.extend(y.numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config_cnn  = load_config('configs/config_cnn.yaml')
    config_pinn = load_config('configs/config_pinn.yaml')

    # --- Antrenament pe date_protocol_clean (leave-segment-out) ---
    data_path = os.path.join(BASE, config_cnn['data_path'])
    train_loader, test_loader, scaler = get_dataloaders(
        data_path, config_cnn['time_steps'], config_cnn['batch_size']
    )

    # --- Test extern: date_vtem resamplat la 20ms ---
    ext_path = os.path.join(BASE, config_cnn['external_test_path'])
    ext_loader = get_external_test_loader(
        ext_path,
        time_steps=config_cnn['time_steps'],
        batch_size=config_cnn['batch_size'],
        scaler=scaler,
        source_dt_ms=config_cnn['external_source_dt_ms'],
        target_dt_ms=config_cnn['external_target_dt_ms'],
    )

    # --- Experiment scarcity pe datele de protocol ---
    fractions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    SIGMA = 0.2
    results = run_scarcity_experiment(
        train_loader.dataset, test_loader, config_cnn, config_pinn, device, fractions,
        n_seeds=7, test_sigma=SIGMA, train_sigma=SIGMA
    )

    print("\n" + "=" * 60)
    print("REZULTATE — TEST PROTOCOL (leave-segment-out)")
    print("=" * 60)
    for i, frac in enumerate(fractions):
        cnn_f1  = results['cnn'][i]
        pinn_f1 = results['pinn'][i]
        winner  = "PINN ✓" if pinn_f1 > cnn_f1 else "CNN ✓"
        print(f"  {frac*100:5.0f}% date: CNN={cnn_f1:.4f}  PINN={pinn_f1:.4f}  -> {winner}")

    # --- Antrenare pe 100% date pentru evaluarea pe vtem extern ---
    print("\n" + "=" * 60)
    print("GENERALIZARE CROSS-DATASET — TEST VTEM (date nevazute)")
    print("=" * 60)
    cnn_model  = train_cnn(train_loader, config_cnn, device)
    pinn_model = train_pinn(train_loader, config_pinn, device)

    cnn_ext  = evaluate_on_loader(cnn_model,  ext_loader, device, is_pinn=False)
    pinn_ext = evaluate_on_loader(pinn_model, ext_loader, device, is_pinn=True)
    winner   = "PINN ✓" if pinn_ext > cnn_ext else "CNN ✓"
    print(f"  CNN  F1 pe vtem: {cnn_ext:.4f}")
    print(f"  PINN F1 pe vtem: {pinn_ext:.4f}  -> {winner}")
    print(f"\n  {'PINN generalizeaza mai bine' if pinn_ext > cnn_ext else 'CNN generalizeaza mai bine'} "
          f"pe setpoint-uri nevazute (+{abs(pinn_ext - cnn_ext):.4f} F1)")

    # --- Salvare modele ---
    save_models(cnn_model, pinn_model, scaler, config_cnn, metrics={
        'cnn_f1_protocol':  float(results['cnn'][0]),   # la 100% date
        'pinn_f1_protocol': float(results['pinn'][0]),
        'cnn_f1_vtem':      float(cnn_ext),
        'pinn_f1_vtem':     float(pinn_ext),
    })


if __name__ == "__main__":
    main()
