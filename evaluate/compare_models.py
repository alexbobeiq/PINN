import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from models.pinn_model import PINN_Classifier
from train.train_data_driven import train_cnn
from train.train_pinn import train_pinn

FAULT_NAMES = {
    0: 'Normal', 1: 'Scurg.ext.V1', 2: 'Scurg.int.V1↔V2',
    3: 'Pre redus\n(1650)', 4: 'Pre ridicat\n(4000)', 5: 'Ex restr.\n(400)',
    6: 'Scurg.ext.V3', 7: 'Scurg.ext.V4'
}
NUM_CLASSES = 8


def _evaluate(model, loader, device, is_pinn, sigma=0.0):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, _, y in loader:
            x = x.to(device)
            if sigma > 0:
                x = x + torch.randn_like(x) * sigma
            logits = model(x)[0] if is_pinn else model(x)
            y_true.extend(y.numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    labels = list(range(NUM_CLASSES))
    return {
        'f1':        f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_per_class': f1_score(y_true, y_pred, average=None,
                                 labels=labels, zero_division=0),
    }


def _train_with_noise(loader, config, device, is_pinn, train_sigma):
    """Wrapper care adauga zgomot la antrenament daca train_sigma > 0."""
    if train_sigma == 0.0:
        return train_pinn(loader, config, device) if is_pinn else train_cnn(loader, config, device)

    class NoisyLoader:
        def __init__(self, base_loader, sigma):
            self.base = base_loader
            self.sigma = sigma
            self.dataset    = base_loader.dataset
            self.batch_size = base_loader.batch_size

        def __iter__(self):
            for x, sp, y in self.base:
                yield x + torch.randn_like(x) * self.sigma, sp, y

        def __len__(self):
            return len(self.base)

    noisy = NoisyLoader(loader, train_sigma)
    return train_pinn(noisy, config, device) if is_pinn else train_cnn(noisy, config, device)


def run_scarcity_experiment(full_train_subset, test_loader, config_cnn, config_pinn, device,
                            fractions=None, n_seeds=7, test_sigma=0.0, train_sigma=None):
    if fractions is None:
        fractions = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    # Daca train_sigma nu e specificat explicit, il egalezi cu test_sigma
    # Asta rezolva overfitting-ul pe distributia curata si curba F1 inversa
    if train_sigma is None:
        train_sigma = test_sigma

    keys = ['f1', 'precision', 'recall']
    results = {f'cnn_{k}': [] for k in keys}
    results.update({f'pinn_{k}': [] for k in keys})
    results.update({f'cnn_{k}_std': [] for k in keys})
    results.update({f'pinn_{k}_std': [] for k in keys})
    results['cnn_perclass']  = []
    results['pinn_perclass'] = []

    actual_dataset = full_train_subset.dataset
    indices = full_train_subset.indices
    train_labels = np.array([actual_dataset.labels[idx + actual_dataset.time_steps]
                              for idx in indices])

    for frac in fractions:
        print(f"\n{'='*50}")
        print(f"  TEST: {frac*100:.0f}% DIN DATE ({int(len(indices)*frac)} samples)")
        print(f"  train_sigma={train_sigma:.2f}  test_sigma={test_sigma:.2f}")
        print(f"{'='*50}")

        cnn_metrics  = {k: [] for k in keys + ['f1_per_class']}
        pinn_metrics = {k: [] for k in keys + ['f1_per_class']}

        for seed in range(n_seeds):
            print(f"\n  --- Seed {seed+1}/{n_seeds} ---")

            if frac < 1.0:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=frac,
                                             random_state=42 + seed)
                local_idx, _ = next(sss.split(np.zeros(len(train_labels)), train_labels))
                final_indices = [indices[i] for i in local_idx]
                train_subset = Subset(actual_dataset, final_indices)
            else:
                train_subset = full_train_subset

            _pin = torch.cuda.is_available()
            _nw  = min(4, (os.cpu_count() or 1) // 2)
            loader = DataLoader(train_subset, batch_size=config_cnn['batch_size'], shuffle=True,
                                num_workers=_nw, pin_memory=_pin, persistent_workers=_nw > 0)

            print(f"  >> CNN ({config_cnn['epochs']} epoci)...")
            cnn_model = _train_with_noise(loader, config_cnn, device, False, train_sigma)
            cm = _evaluate(cnn_model, test_loader, device, False, test_sigma)
            for k in keys:
                cnn_metrics[k].append(cm[k])
            cnn_metrics['f1_per_class'].append(cm['f1_per_class'])
            print(f"  >> CNN  F1={cm['f1']:.4f}  P={cm['precision']:.4f}  R={cm['recall']:.4f}")

            print(f"  >> PINN ({config_pinn['epochs']} epoci)...")
            pinn_model = _train_with_noise(loader, config_pinn, device, True, train_sigma)
            pm = _evaluate(pinn_model, test_loader, device, True, test_sigma)
            for k in keys:
                pinn_metrics[k].append(pm[k])
            pinn_metrics['f1_per_class'].append(pm['f1_per_class'])
            print(f"  >> PINN F1={pm['f1']:.4f}  P={pm['precision']:.4f}  R={pm['recall']:.4f}")

        for k in keys:
            results[f'cnn_{k}'].append(np.mean(cnn_metrics[k]))
            results[f'pinn_{k}'].append(np.mean(pinn_metrics[k]))
            results[f'cnn_{k}_std'].append(np.std(cnn_metrics[k]))
            results[f'pinn_{k}_std'].append(np.std(pinn_metrics[k]))

        results['cnn_perclass'].append(np.mean(cnn_metrics['f1_per_class'],  axis=0))
        results['pinn_perclass'].append(np.mean(pinn_metrics['f1_per_class'], axis=0))

        winner = "PINN" if results['pinn_f1'][-1] > results['cnn_f1'][-1] else "CNN"
        print(f"\n  >> REZULTAT {frac*100:.0f}%:  "
              f"CNN={results['cnn_f1'][-1]:.4f}±{results['cnn_f1_std'][-1]:.4f}  "
              f"PINN={results['pinn_f1'][-1]:.4f}±{results['pinn_f1_std'][-1]:.4f}  "
              f"→ {winner}")

    plot_results(fractions, results, test_sigma, train_sigma)
    return results


def plot_results(fractions, results, test_sigma=0.0, train_sigma=0.0):
    x = [f * 100 for f in fractions]
    noise_str = (f"Train σ={train_sigma} | Test σ={test_sigma}"
                 if test_sigma > 0 else "Date curate")

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'PINN vs CNN — {noise_str}', fontsize=14, fontweight='bold')

    colors = {'cnn': 'red', 'pinn': 'green'}
    markers = {'cnn': 'o', 'pinn': 's'}
    ls = {'cnn': '--', 'pinn': '-'}

    # ── 1. F1 Weighted ──────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    for m, label in [('cnn', 'CNN (Statistic)'), ('pinn', 'PINN (Fizică)')]:
        mean = np.array(results[f'{m}_f1'])
        std  = np.array(results[f'{m}_f1_std'])
        ax1.plot(x, mean, f"{markers[m]}{ls[m]}", color=colors[m], label=label, markersize=7)
        ax1.fill_between(x, mean-std, mean+std, alpha=0.15, color=colors[m])
    ax1.set_title('F1 Weighted')
    ax1.set_xlabel('% Date antrenament'); ax1.set_ylabel('F1')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3); ax1.invert_xaxis()

    # ── 2. Precision Weighted ────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    for m, label in [('cnn', 'CNN'), ('pinn', 'PINN')]:
        mean = np.array(results[f'{m}_precision'])
        std  = np.array(results[f'{m}_precision_std'])
        ax2.plot(x, mean, f"{markers[m]}{ls[m]}", color=colors[m], label=label, markersize=7)
        ax2.fill_between(x, mean-std, mean+std, alpha=0.15, color=colors[m])
    ax2.set_title('Precision Weighted')
    ax2.set_xlabel('% Date antrenament'); ax2.set_ylabel('Precision')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3); ax2.invert_xaxis()

    # ── 3. Recall Weighted ───────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    for m, label in [('cnn', 'CNN'), ('pinn', 'PINN')]:
        mean = np.array(results[f'{m}_recall'])
        std  = np.array(results[f'{m}_recall_std'])
        ax3.plot(x, mean, f"{markers[m]}{ls[m]}", color=colors[m], label=label, markersize=7)
        ax3.fill_between(x, mean-std, mean+std, alpha=0.15, color=colors[m])
    ax3.set_title('Recall Weighted')
    ax3.set_xlabel('% Date antrenament'); ax3.set_ylabel('Recall')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3); ax3.invert_xaxis()

    # ── 4. Avantaj PINN-CNN (ΔF1) ────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    delta = np.array(results['pinn_f1']) - np.array(results['cnn_f1'])
    delta_std = np.sqrt(np.array(results['pinn_f1_std'])**2 +
                        np.array(results['cnn_f1_std'])**2)
    bar_colors = ['green' if d > 0 else 'red' for d in delta]
    bars = ax4.bar(range(len(x)), delta, color=bar_colors, alpha=0.7, width=0.5)
    ax4.errorbar(range(len(x)), delta, yerr=delta_std, fmt='none', color='black', capsize=4)
    ax4.set_xticks(range(len(x)))
    ax4.set_xticklabels([f'{v:.0f}%' for v in x])
    ax4.axhline(0, color='black', linewidth=0.8)
    ax4.set_title('Avantaj PINN față de CNN (ΔF1)\n(verde = PINN câștigă)')
    ax4.set_xlabel('% Date antrenament'); ax4.set_ylabel('PINN F1 − CNN F1')
    ax4.grid(True, alpha=0.3, axis='y')

    # ── 5. F1 per clasă la fracția cu cel mai mare avantaj PINN ─
    ax5 = fig.add_subplot(2, 3, 5)
    best_frac_idx = int(np.argmax(delta))
    cnn_pc  = results['cnn_perclass'][best_frac_idx]
    pinn_pc = results['pinn_perclass'][best_frac_idx]
    class_names = [FAULT_NAMES.get(i, f'C{i}') for i in range(NUM_CLASSES)]
    xc = np.arange(NUM_CLASSES)
    w  = 0.35
    ax5.bar(xc - w/2, cnn_pc,  w, label='CNN',  color='red',   alpha=0.7)
    ax5.bar(xc + w/2, pinn_pc, w, label='PINN', color='green', alpha=0.7)
    ax5.set_xticks(xc)
    ax5.set_xticklabels(class_names, fontsize=7, rotation=15, ha='right')
    ax5.set_title(f'F1 per clasă la {x[best_frac_idx]:.0f}% date\n(fracția cu max avantaj PINN)')
    ax5.set_ylabel('F1'); ax5.set_ylim([0, 1.05])
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3, axis='y')

    # ── 6. Tabel rezumat ─────────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    col_labels = ['Date%', 'CNN F1', 'PINN F1', 'ΔPINN', 'CNN P', 'PINN P']
    rows = []
    for i, frac in enumerate(fractions):
        delta_i = results['pinn_f1'][i] - results['cnn_f1'][i]
        sign = '+' if delta_i >= 0 else ''
        rows.append([
            f"{frac*100:.0f}%",
            f"{results['cnn_f1'][i]:.4f}",
            f"{results['pinn_f1'][i]:.4f}",
            f"{sign}{delta_i:.4f}",
            f"{results['cnn_precision'][i]:.4f}",
            f"{results['pinn_precision'][i]:.4f}",
        ])
    tbl = ax6.table(cellText=rows, colLabels=col_labels,
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#404040')
            cell.set_text_props(color='white', fontweight='bold')
        elif c == 3:
            val = rows[r-1][3] if r > 0 else ''
            cell.set_facecolor('#d4f5d4' if (val and val[0] != '-') else '#f5d4d4')
    ax6.set_title('Rezumat numeric', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/results_comparison.png', dpi=150)
    print("\n[INFO] Grafic salvat: results/results_comparison.png")
    plt.show()
