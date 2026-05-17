import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from train.train_data_driven import train_cnn
from train.train_pinn import train_pinn

NUM_CLASSES = 8
FAULT_NAMES = {
    0: 'Normal', 1: 'Scurg.ext\nV1', 2: 'Scurg.int\nV1↔V2',
    3: 'Pre redus\n(1650)', 4: 'Pre ridicat\n(4000)', 5: 'Ex restr.\n(400)',
    6: 'Scurg.ext\nV3', 7: 'Scurg.ext\nV4'
}


def _evaluate_noisy(model, loader, device, sigma, is_pinn):
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
        'f1_class':  f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0),
    }


def _train_noisy(loader, config, device, is_pinn, train_sigma):
    class NoisyLoader:
        def __init__(self, base, sigma):
            self.base = base; self.sigma = sigma
            self.dataset = base.dataset; self.batch_size = base.batch_size
        def __iter__(self):
            for x, sp, y in self.base:
                yield x + torch.randn_like(x) * self.sigma, sp, y
        def __len__(self): return len(self.base)

    src = NoisyLoader(loader, train_sigma) if train_sigma > 0 else loader
    return train_pinn(src, config, device) if is_pinn else train_cnn(src, config, device)


def run_noise_experiment(train_loader, test_loader, config_cnn, config_pinn, device,
                         noise_levels=None, n_seeds=5, train_sigma=0.0,
                         train_fraction=1.0, pretrained_cnn=None, pretrained_pinn=None):
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    # Subset de antrenament daca train_fraction < 1
    base_dataset  = train_loader.dataset
    base_indices  = np.array(base_dataset.indices
                             if hasattr(base_dataset, 'indices') else range(len(base_dataset)))
    base_labels   = np.array([base_dataset.dataset.labels[i + base_dataset.dataset.time_steps]
                               for i in base_indices]) if train_fraction < 1.0 else None

    using_pretrained = pretrained_cnn is not None and pretrained_pinn is not None

    print(f"\n{'='*55}")
    print(f"  NOISE ROBUSTNESS  train_sigma={train_sigma}  "
          f"frac={train_fraction*100:.0f}%  "
          f"{'pretrained' if using_pretrained else f'seeds={n_seeds}'}")
    print(f"{'='*55}")

    if using_pretrained:
        # Folosim modelele salvate direct — fara reantrenament
        print("\n[INFO] Folosind modele pre-antrenate din checkpoint.")
        cnn_models  = [pretrained_cnn]
        pinn_models = [pretrained_pinn]
    else:
        # Antrenam modele o singura data per seed (pe nivelul de train_sigma ales)
        print(f"\nAntrenare {n_seeds} perechi CNN/PINN...")
        cnn_models, pinn_models = [], []
        for seed in range(n_seeds):
            torch.manual_seed(42 + seed)
            if train_fraction < 1.0:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=train_fraction,
                                            random_state=42 + seed)
                idx, _ = next(sss.split(np.zeros(len(base_labels)), base_labels))
                sub = Subset(base_dataset, [base_indices[i] for i in idx])
                ldr = DataLoader(sub, batch_size=config_cnn['batch_size'], shuffle=True)
            else:
                ldr = train_loader

            print(f"  Seed {seed+1}/{n_seeds}  CNN...", end='  ')
            cnn_models.append(_train_noisy(ldr, config_cnn, device, False, train_sigma))
            print(f"PINN...")
            pinn_models.append(_train_noisy(ldr, config_pinn, device, True, train_sigma))

    # Evaluam la fiecare nivel de zgomot
    res = {k: [] for k in ['cnn_f1', 'pinn_f1', 'cnn_f1_std', 'pinn_f1_std',
                            'cnn_prec', 'pinn_prec', 'cnn_rec', 'pinn_rec',
                            'cnn_class', 'pinn_class']}

    print(f"\nEvaluare la {len(noise_levels)} niveluri de zgomot...")
    for sigma in noise_levels:
        cnn_m  = [_evaluate_noisy(m, test_loader, device, sigma, False) for m in cnn_models]
        pinn_m = [_evaluate_noisy(m, test_loader, device, sigma, True)  for m in pinn_models]

        res['cnn_f1'].append(np.mean([m['f1'] for m in cnn_m]))
        res['pinn_f1'].append(np.mean([m['f1'] for m in pinn_m]))
        res['cnn_f1_std'].append(np.std([m['f1'] for m in cnn_m]))
        res['pinn_f1_std'].append(np.std([m['f1'] for m in pinn_m]))
        res['cnn_prec'].append(np.mean([m['precision'] for m in cnn_m]))
        res['pinn_prec'].append(np.mean([m['precision'] for m in pinn_m]))
        res['cnn_rec'].append(np.mean([m['recall'] for m in cnn_m]))
        res['pinn_rec'].append(np.mean([m['recall'] for m in pinn_m]))
        res['cnn_class'].append(np.mean([m['f1_class'] for m in cnn_m], axis=0))
        res['pinn_class'].append(np.mean([m['f1_class'] for m in pinn_m], axis=0))

        winner = "PINN" if res['pinn_f1'][-1] > res['cnn_f1'][-1] else "CNN "
        delta  = res['pinn_f1'][-1] - res['cnn_f1'][-1]
        print(f"  σ={sigma:.2f}:  CNN={res['cnn_f1'][-1]:.4f}  "
              f"PINN={res['pinn_f1'][-1]:.4f}  Δ={delta:+.4f}  → {winner}")

    _plot(noise_levels, res, train_sigma, train_fraction)
    return res


def _plot(noise_levels, res, train_sigma, train_fraction):
    nl = np.array(noise_levels)
    cnn_f1   = np.array(res['cnn_f1']);   pinn_f1   = np.array(res['pinn_f1'])
    cnn_std  = np.array(res['cnn_f1_std']); pinn_std = np.array(res['pinn_f1_std'])
    delta    = pinn_f1 - cnn_f1

    title = (f"Robustețe la Zgomot: PINN vs CNN\n"
             f"Antrenament: σ={train_sigma}, {train_fraction*100:.0f}% date")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # ── 1. F1 vs sigma ────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(nl, cnn_f1,  'o--', color='red',   label='CNN (Statistic)', markersize=7, lw=2)
    ax.fill_between(nl, cnn_f1-cnn_std, cnn_f1+cnn_std, alpha=0.15, color='red')
    ax.plot(nl, pinn_f1, 's-',  color='green', label='PINN (Fizică)',   markersize=7, lw=2)
    ax.fill_between(nl, pinn_f1-pinn_std, pinn_f1+pinn_std, alpha=0.15, color='green')
    # Marcam punctul de train_sigma
    if train_sigma in noise_levels:
        idx = noise_levels.index(train_sigma)
        ax.axvline(train_sigma, color='gray', linestyle=':', alpha=0.7,
                   label=f'σ antrenament ({train_sigma})')
    ax.set_xlabel('σ zgomot test'); ax.set_ylabel('F1 Weighted')
    ax.set_title('F1 Weighted vs Zgomot'); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3); ax.set_ylim([0, 1.05])

    # ── 2. Precision si Recall vs sigma ───────────────────────
    ax = axes[0, 1]
    ax.plot(nl, res['cnn_prec'],  'o--', color='red',   label='CNN Precision', markersize=6, lw=1.5)
    ax.plot(nl, res['pinn_prec'], 's-',  color='green', label='PINN Precision', markersize=6, lw=1.5)
    ax.plot(nl, res['cnn_rec'],   '^--', color='darkred',   label='CNN Recall', markersize=6, lw=1.5, alpha=0.7)
    ax.plot(nl, res['pinn_rec'],  'D-',  color='darkgreen', label='PINN Recall', markersize=6, lw=1.5, alpha=0.7)
    ax.set_xlabel('σ zgomot test'); ax.set_ylabel('Scor')
    ax.set_title('Precision & Recall vs Zgomot'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3); ax.set_ylim([0, 1.05])

    # ── 3. Avantaj PINN (ΔF1) vs sigma ───────────────────────
    ax = axes[0, 2]
    ax.fill_between(nl, 0, delta, where=delta >= 0, alpha=0.4, color='green', label='PINN câștigă')
    ax.fill_between(nl, 0, delta, where=delta < 0,  alpha=0.4, color='red',   label='CNN câștigă')
    ax.plot(nl, delta, 'k-o', markersize=6, lw=2, label='ΔF1 = PINN − CNN')
    ax.axhline(0, color='black', lw=0.8)
    # Crossover point
    crossover = None
    for i in range(len(delta)-1):
        if delta[i] <= 0 and delta[i+1] > 0:
            crossover = nl[i] + (nl[i+1]-nl[i]) * (-delta[i])/(delta[i+1]-delta[i])
    if crossover:
        ax.axvline(crossover, color='orange', linestyle='--', alpha=0.8,
                   label=f'Crossover σ≈{crossover:.2f}')
    ax.set_xlabel('σ zgomot test'); ax.set_ylabel('PINN F1 − CNN F1')
    ax.set_title('Avantaj PINN față de CNN'); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 4. Degradare relativa (%) fata de sigma=0 ─────────────
    ax = axes[1, 0]
    cnn_drop  = (cnn_f1[0]  - cnn_f1)  / (cnn_f1[0]  + 1e-9) * 100
    pinn_drop = (pinn_f1[0] - pinn_f1) / (pinn_f1[0] + 1e-9) * 100
    ax.plot(nl, cnn_drop,  'o--', color='red',   label='CNN', markersize=7, lw=2)
    ax.plot(nl, pinn_drop, 's-',  color='green', label='PINN', markersize=7, lw=2)
    ax.set_xlabel('σ zgomot test'); ax.set_ylabel('Degradare F1 (%)')
    ax.set_title('Degradare relativă față de σ=0\n(mai mic = mai robust)')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5)

    # ── 5. Heatmap F1 per clasa vs sigma (PINN - CNN) ─────────
    ax = axes[1, 1]
    cnn_cls  = np.array(res['cnn_class'])   # [n_sigma, n_classes]
    pinn_cls = np.array(res['pinn_class'])
    diff_cls = pinn_cls - cnn_cls            # pozitiv = PINN mai bun

    im = ax.imshow(diff_cls.T, aspect='auto', cmap='RdYlGn',
                   vmin=-0.1, vmax=0.1,
                   extent=[nl[0], nl[-1], -0.5, NUM_CLASSES-0.5])
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels([FAULT_NAMES.get(i, f'C{i}').replace('\n', ' ')
                        for i in range(NUM_CLASSES)], fontsize=8)
    ax.set_xlabel('σ zgomot test')
    ax.set_title('PINN − CNN per clasă (verde=PINN mai bun)')
    plt.colorbar(im, ax=ax, label='ΔF1')

    # ── 6. F1 per clasa la sigma optim (max avantaj PINN) ─────
    ax = axes[1, 2]
    best_sigma_idx = int(np.argmax(delta))
    cnn_bc  = res['cnn_class'][best_sigma_idx]
    pinn_bc = res['pinn_class'][best_sigma_idx]
    xc = np.arange(NUM_CLASSES); w = 0.35
    ax.bar(xc - w/2, cnn_bc,  w, label='CNN',  color='red',   alpha=0.75)
    ax.bar(xc + w/2, pinn_bc, w, label='PINN', color='green', alpha=0.75)
    ax.set_xticks(xc)
    ax.set_xticklabels([FAULT_NAMES.get(i, f'C{i}').replace('\n', ' ')
                        for i in range(NUM_CLASSES)], fontsize=7, rotation=20, ha='right')
    ax.set_title(f'F1 per clasă la σ={noise_levels[best_sigma_idx]}\n(max avantaj PINN)')
    ax.set_ylabel('F1'); ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/noise_robustness_results.png', dpi=150)
    print("\n[INFO] Grafic salvat: results/noise_robustness_results.png")
    plt.show()
