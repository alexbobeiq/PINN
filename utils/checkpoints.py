import os
import pickle
import torch
from datetime import datetime

from models.pinn_model import PINN_Classifier
from models.data_driven import CNN1D_FaultClassifier

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')


def save_models(cnn_model, pinn_model, scaler, config, metrics=None):
    """Salveaza CNN, PINN si scaler in checkpoints/."""
    os.makedirs(CKPT_DIR, exist_ok=True)

    arch = {
        'num_features':     2,
        'num_classes':      config['num_classes'],
        'time_steps':       config['time_steps'],
        'backbone_channels': config.get('backbone_channels', [32, 64, 128]),
    }
    meta = {
        'arch':     arch,
        'metrics':  metrics or {},
        'saved_at': datetime.now().isoformat(),
        'data_path': config.get('data_path', ''),
    }

    torch.save({'state_dict': cnn_model.state_dict(),  **meta}, os.path.join(CKPT_DIR, 'cnn.pt'))
    torch.save({'state_dict': pinn_model.state_dict(), **meta}, os.path.join(CKPT_DIR, 'pinn.pt'))

    with open(os.path.join(CKPT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n[Checkpoint] Salvat in: {CKPT_DIR}")
    if metrics:
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


def load_models(device):
    """
    Incarca CNN, PINN si scaler din checkpoints/.
    Returneaza (cnn_model, pinn_model, scaler) sau None daca nu exista checkpoint.
    """
    paths = {
        'cnn':    os.path.join(CKPT_DIR, 'cnn.pt'),
        'pinn':   os.path.join(CKPT_DIR, 'pinn.pt'),
        'scaler': os.path.join(CKPT_DIR, 'scaler.pkl'),
    }

    if not all(os.path.exists(p) for p in paths.values()):
        return None

    cnn_ckpt  = torch.load(paths['cnn'],  map_location=device, weights_only=False)
    pinn_ckpt = torch.load(paths['pinn'], map_location=device, weights_only=False)
    arch = cnn_ckpt['arch']

    cnn_model = CNN1D_FaultClassifier(**arch)
    cnn_model.load_state_dict(cnn_ckpt['state_dict'])
    cnn_model.to(device).eval()

    pinn_model = PINN_Classifier(**arch)
    pinn_model.load_state_dict(pinn_ckpt['state_dict'])
    pinn_model.to(device).eval()

    with open(paths['scaler'], 'rb') as f:
        scaler = pickle.load(f)

    print(f"[Checkpoint] Incarcat din: {CKPT_DIR}")
    print(f"  Salvat la:   {cnn_ckpt.get('saved_at', '?')}")
    print(f"  Date:        {cnn_ckpt.get('data_path', '?')}")
    metrics = cnn_ckpt.get('metrics', {})
    if metrics:
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return cnn_model, pinn_model, scaler
