import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset
from train.train_autoencoder import train_autoencoder
from train.train_pinn_anomaly import train_pinn_anomaly

FAULT_NAMES = {
    0: 'Normal',
    1: 'Scurg. ext. V1',
    2: 'Scurg. int. V1↔V2',
    3: 'Pre redus (1650)',
    4: 'Pre ridicat (4000)',
    5: 'Ex restrict. (400)',
    6: 'Scurg. ext. V3',
    7: 'Scurg. ext. V4',
}


def _collect_scores(model, loader, device, is_pinn=False):
    model.eval()
    scores, labels, taus_f, taus_e = [], [], [], []
    with torch.no_grad():
        for x_scaled, _, y in loader:
            x_scaled = x_scaled.to(device)
            s = model.anomaly_score(x_scaled)
            if not isinstance(s, np.ndarray):
                s = s.cpu().numpy()
            scores.extend(s)
            labels.extend(y.numpy())
            if is_pinn:
                tf, te = model.get_tau(x_scaled)
                taus_f.extend(tf.cpu().numpy())
                taus_e.extend(te.cpu().numpy())
    return (np.array(scores), np.array(labels),
            np.array(taus_f) if is_pinn else None,
            np.array(taus_e) if is_pinn else None)


def run_anomaly_experiment(full_train_subset, test_loader, config, device):
    actual_dataset = full_train_subset.dataset
    indices        = np.array(full_train_subset.indices)
    train_labels   = np.array([
        actual_dataset.labels[idx + actual_dataset.time_steps] for idx in indices
    ])

    # Antrenare EXCLUSIV pe clasa 0 (normal)
    class0_idx    = indices[train_labels == 0]
    class0_loader = DataLoader(
        Subset(actual_dataset, class0_idx),
        batch_size=config['batch_size'], shuffle=True
    )
    print(f"\n[INFO] Antrenare pe {len(class0_idx)} ferestre normale (clasa 0)")

    print("\n>> Antrenare PINN Anomaly Detector (ODE loss pe date normale)...")
    pinn_model = train_pinn_anomaly(class0_loader, config, device)

    print("\n>> Antrenare CNN Autoencoder (reconstructie pe date normale)...")
    ae_model = train_autoencoder(class0_loader, config, device)

    print("\n>> Calibrare PINN: calculez distributia tau pe date normale...")
    pinn_model.calibrate(class0_loader, device)

    print("\n>> Calculez scoruri de anomalie pe setul de test...")
    pinn_scores, labels, tau_fill, tau_exhaust = _collect_scores(
        pinn_model, test_loader, device, is_pinn=True)
    ae_scores, _, _, _ = _collect_scores(ae_model, test_loader, device)

    binary_labels = (labels != 0).astype(int)

    pinn_auroc = roc_auc_score(binary_labels, pinn_scores)
    ae_auroc   = roc_auc_score(binary_labels, ae_scores)

    print(f"\n{'='*55}")
    print(f"  AUROC  Normal vs Orice Defect:")
    print(f"    PINN  (rezidual ODE)    : {pinn_auroc:.4f}")
    print(f"    Autoencoder (MSE recon) : {ae_auroc:.4f}")
    print(f"{'='*55}")

    print(f"\n  AUROC per clasa de defect (Normal vs Clasa X):")
    for fault in sorted(np.unique(labels[labels != 0])):
        mask = (labels == 0) | (labels == fault)
        bl   = (labels[mask] != 0).astype(int)
        pf   = roc_auc_score(bl, pinn_scores[mask])
        af   = roc_auc_score(bl, ae_scores[mask])
        name = FAULT_NAMES.get(int(fault), f'C{fault}')
        winner = 'PINN ✓' if pf > af else 'AE  ✓'
        print(f"    Clasa {int(fault)} {name:<22}: PINN={pf:.4f}  AE={af:.4f}  → {winner}")

    _plot(pinn_scores, ae_scores, labels, tau_fill, tau_exhaust, binary_labels)
    return pinn_scores, ae_scores, labels


def _plot(pinn_scores, ae_scores, labels, tau_fill, tau_exhaust, binary_labels):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detecție Anomalii: PINN vs Autoencoder CNN', fontsize=14, fontweight='bold')

    unique_labels = np.array(sorted(np.unique(labels)))
    class_names   = [FAULT_NAMES.get(int(c), f'C{c}') for c in unique_labels]
    colors        = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    # ── 1. Curbe ROC ──────────────────────────────────────────
    ax = axes[0, 0]
    for scores, name, color in [
        (pinn_scores, 'PINN (rezidual ODE)', 'green'),
        (ae_scores,   'Autoencoder CNN',     'red'),
    ]:
        fpr, tpr, _ = roc_curve(binary_labels, scores)
        auroc = roc_auc_score(binary_labels, scores)
        ax.plot(fpr, tpr, label=f'{name}  AUROC={auroc:.3f}', color=color, lw=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC  —  Normal vs Orice Defect')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 2. Box plot scoruri PINN per clasă ────────────────────
    ax = axes[0, 1]
    data = [pinn_scores[labels == c] for c in unique_labels]
    bp   = ax.boxplot(data, labels=class_names, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor('lightgreen' if unique_labels[i] == 0 else 'salmon')
    ax.set_title('PINN: Rezidual ODE per clasă')
    ax.set_ylabel('Scor anomalie (mai mare = mai anormal)')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3)

    # ── 3. Box plot scoruri AE per clasă ─────────────────────
    ax = axes[1, 0]
    data_ae = [ae_scores[labels == c] for c in unique_labels]
    bp2     = ax.boxplot(data_ae, labels=class_names, patch_artist=True)
    for i, patch in enumerate(bp2['boxes']):
        patch.set_facecolor('lightgreen' if unique_labels[i] == 0 else 'salmon')
    ax.set_title('Autoencoder: Eroare reconstrucție per clasă')
    ax.set_ylabel('MSE reconstrucție')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3)

    # ── 4. Scatter τ_fill vs τ_exhaust per clasă ─────────────
    ax = axes[1, 1]
    for i, c in enumerate(unique_labels):
        mask = labels == c
        ax.scatter(tau_fill[mask], tau_exhaust[mask],
                   alpha=0.25, s=8, color=colors[i],
                   label=FAULT_NAMES.get(int(c), f'C{c}'))
    ax.set_xlabel('τ_fill  (constantă umplere)')
    ax.set_ylabel('τ_exhaust  (constantă evacuare)')
    ax.set_title('PINN: Parametri fizici per clasă\n(interpretabilitate τ)')
    ax.legend(fontsize=7, markerscale=3, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/anomaly_detection_results.png', dpi=150)
    print("\n[INFO] Grafic salvat: results/anomaly_detection_results.png")
    plt.show()
