import os
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
    6: 'Intarziere retras (1s)',
    7: 'Eroare senzor (+500)',
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
    # label for window i = labels at last row of window = valid_indices[i] + T - 1
    train_labels   = np.array([
        actual_dataset.labels[actual_dataset.valid_indices[idx] + actual_dataset.time_steps - 1]
        for idx in indices
    ])

    _pin = torch.cuda.is_available()
    _nw  = min(4, (os.cpu_count() or 1) // 2)

    # Antrenare EXCLUSIV pe clasa 0 (normal)
    class0_idx    = indices[train_labels == 0]
    class0_loader = DataLoader(
        Subset(actual_dataset, class0_idx),
        batch_size=config['batch_size'], shuffle=True,
        num_workers=_nw, pin_memory=_pin, persistent_workers=_nw > 0,
    )
    print(f"\n[INFO] Antrenare pe {len(class0_idx)} ferestre normale (clasa 0)")

    print("\n>> Antrenare PINN Anomaly Detector (ODE loss pe date normale)...")
    pinn_model = train_pinn_anomaly(class0_loader, config, device)

    print("\n>> Antrenare CNN Autoencoder (reconstructie pe date normale)...")
    ae_model = train_autoencoder(class0_loader, config, device)

    print("\n>> Calibrare PINN (mode=combined — 9 features derivate din ODE)...")
    pinn_model.calibrate(class0_loader, device, mode='combined')

    print("\n>> Calculez scoruri de anomalie pe setul de test...")
    pinn_scores, labels, tau_fill, tau_exhaust = _collect_scores(
        pinn_model, test_loader, device, is_pinn=True)
    ae_scores, _, _, _ = _collect_scores(ae_model, test_loader, device)

    pinn_scores_viz = np.log1p(np.clip(pinn_scores, 0, np.percentile(pinn_scores, 99)))

    binary_labels = (labels != 0).astype(int)

    pinn_auroc = roc_auc_score(binary_labels, pinn_scores)
    ae_auroc   = roc_auc_score(binary_labels, ae_scores)

    print(f"\n{'='*65}")
    print(f"  AUROC  Normal vs Orice Defect:")
    print(f"    PINN [9 features ODE: τ, P_eq, amp, frac, res] : {pinn_auroc:.4f}")
    print(f"    Autoencoder CNN [MSE reconstructie semnal]      : {ae_auroc:.4f}")
    print(f"{'='*65}")

    print(f"\n  AUROC per clasa de defect (Normal vs Clasa X):")
    for fault in sorted(np.unique(labels[labels != 0])):
        mask = (labels == 0) | (labels == fault)
        bl   = (labels[mask] != 0).astype(int)
        pf   = roc_auc_score(bl, pinn_scores[mask])
        af   = roc_auc_score(bl, ae_scores[mask])
        name = FAULT_NAMES.get(int(fault), f'C{fault}')
        winner = 'PINN ✓' if pf > af else 'AE  ✓'
        print(f"    Clasa {int(fault)} {name:<25}: PINN={pf:.4f}  AE={af:.4f}  → {winner}")

    _plot(pinn_scores_viz, ae_scores, labels, tau_fill, tau_exhaust, binary_labels)
    return pinn_scores, ae_scores, labels


def run_anomaly_noise_experiment(full_train_subset, test_loader, config, device,
                                 noise_levels=None, n_seeds=3):
    """
    Noise robustness pe anomaly detection: PINN-AD vs AE.
    Ambele antrenate pe date Normale CURATE; evaluate cu zgomot Gaussian crescator.
    Argumentul: features fizice (τ, ode_res) sunt mai stabile la zgomot
    decat reconstructia bruta a AE-ului.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    actual_dataset = full_train_subset.dataset
    indices        = np.array(full_train_subset.indices)
    train_labels   = np.array([
        actual_dataset.labels[actual_dataset.valid_indices[idx] + actual_dataset.time_steps - 1]
        for idx in indices
    ])
    class0_idx = indices[train_labels == 0]

    _pin = torch.cuda.is_available()
    _nw  = min(4, (os.cpu_count() or 1) // 2)

    binary_labels = np.array([
        (actual_dataset.labels[actual_dataset.valid_indices[idx] + actual_dataset.time_steps - 1] != 0)
        for idx in np.array(test_loader.dataset.indices)
    ], dtype=int)

    pinn_auroc_mean, pinn_auroc_std = [], []
    ae_auroc_mean,   ae_auroc_std   = [], []

    # Antreneaza n_seeds perechi pe date curate
    print(f"Antrenare {n_seeds} perechi PINN-AD / AE pe date Normale curate...")
    pinn_models, ae_models = [], []
    for seed in range(n_seeds):
        rng    = np.random.default_rng(seed)
        sub    = rng.choice(class0_idx, size=len(class0_idx), replace=False)
        loader = DataLoader(
            Subset(actual_dataset, sub),
            batch_size=config['batch_size'], shuffle=True,
            num_workers=_nw, pin_memory=_pin, persistent_workers=_nw > 0,
        )
        print(f"  Seed {seed+1}/{n_seeds}  PINN-AD...", end='  ')
        pinn = train_pinn_anomaly(loader, config, device)
        pinn.calibrate(loader, device, mode='combined')
        pinn_models.append(pinn)
        print("AE...")
        ae_models.append(train_autoencoder(loader, config, device))

    # Evalueaza la fiecare nivel de zgomot
    print(f"\nEvaluare la {len(noise_levels)} niveluri de zgomot...")
    for sigma in noise_levels:
        pinn_runs, ae_runs = [], []
        for pinn, ae in zip(pinn_models, ae_models):
            ps,  _, _, _ = _collect_scores_noisy(pinn, test_loader, device, sigma, is_pinn=True)
            aes, _, _, _ = _collect_scores_noisy(ae,   test_loader, device, sigma)
            pinn_runs.append(roc_auc_score(binary_labels, ps))
            ae_runs.append(roc_auc_score(binary_labels, aes))

        pinn_auroc_mean.append(np.mean(pinn_runs))
        pinn_auroc_std.append(np.std(pinn_runs))
        ae_auroc_mean.append(np.mean(ae_runs))
        ae_auroc_std.append(np.std(ae_runs))

        winner = "PINN" if pinn_auroc_mean[-1] > ae_auroc_mean[-1] else "AE  "
        delta  = pinn_auroc_mean[-1] - ae_auroc_mean[-1]
        print(f"  σ={sigma:.2f}:  PINN-AD={pinn_auroc_mean[-1]:.4f}  "
              f"AE={ae_auroc_mean[-1]:.4f}  Δ={delta:+.4f}  → {winner}")

    _plot_noise(noise_levels, pinn_auroc_mean, pinn_auroc_std,
                ae_auroc_mean, ae_auroc_std)
    return {
        'noise_levels':    noise_levels,
        'pinn_auroc_mean': pinn_auroc_mean,
        'pinn_auroc_std':  pinn_auroc_std,
        'ae_auroc_mean':   ae_auroc_mean,
        'ae_auroc_std':    ae_auroc_std,
    }


def _collect_scores_noisy(model, loader, device, sigma, is_pinn=False):
    """Ca _collect_scores dar adauga zgomot Gaussian pe input."""
    model.eval()
    scores, labels, taus_f, taus_e = [], [], [], []
    with torch.no_grad():
        for x_scaled, _, y in loader:
            x_scaled = x_scaled.to(device)
            if sigma > 0:
                x_scaled = x_scaled + torch.randn_like(x_scaled) * sigma
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


def _plot_noise(noise_levels, pinn_mean, pinn_std, ae_mean, ae_std):
    nl = np.array(noise_levels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Robustețe la Zgomot — Anomaly Detection: PINN vs Autoencoder\n'
                 'Antrenament pe date Normale curate; evaluare cu zgomot crescător',
                 fontsize=12, fontweight='bold')

    # ── 1. AUROC vs sigma ─────────────────────────────────────
    ax = axes[0]
    ax.plot(nl, pinn_mean, 's-',  color='green', label='PINN-AD (features ODE)', lw=2, markersize=7)
    ax.fill_between(nl, np.array(pinn_mean)-np.array(pinn_std),
                        np.array(pinn_mean)+np.array(pinn_std), alpha=0.15, color='green')
    ax.plot(nl, ae_mean,   'o--', color='red',   label='Autoencoder CNN',        lw=2, markersize=7)
    ax.fill_between(nl, np.array(ae_mean)-np.array(ae_std),
                        np.array(ae_mean)+np.array(ae_std), alpha=0.15, color='red')
    ax.set_xlabel('σ zgomot test'); ax.set_ylabel('AUROC')
    ax.set_title('AUROC vs Zgomot'); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3); ax.set_ylim(0.4, 1.05)

    # ── 2. Avantaj PINN (ΔAUROC) ──────────────────────────────
    ax = axes[1]
    delta = np.array(pinn_mean) - np.array(ae_mean)
    ax.fill_between(nl, 0, delta, where=delta >= 0, alpha=0.4, color='green', label='PINN câștigă')
    ax.fill_between(nl, 0, delta, where=delta <  0, alpha=0.4, color='red',   label='AE câștigă')
    ax.plot(nl, delta, 'k-o', markersize=6, lw=2)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('σ zgomot test'); ax.set_ylabel('PINN AUROC − AE AUROC')
    ax.set_title('Avantaj PINN față de AE'); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 3. Degradare relativa fata de sigma=0 ─────────────────
    ax = axes[2]
    pinn_drop = (pinn_mean[0] - np.array(pinn_mean)) / (pinn_mean[0] + 1e-9) * 100
    ae_drop   = (ae_mean[0]   - np.array(ae_mean))   / (ae_mean[0]   + 1e-9) * 100
    ax.plot(nl, pinn_drop, 's-',  color='green', label='PINN-AD', lw=2, markersize=7)
    ax.plot(nl, ae_drop,   'o--', color='red',   label='AE CNN',  lw=2, markersize=7)
    ax.set_xlabel('σ zgomot test'); ax.set_ylabel('Degradare AUROC (%)')
    ax.set_title('Degradare relativă față de σ=0\n(mai mic = mai robust)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5)

    plt.tight_layout()
    plt.savefig('results/anomaly_noise_results.png', dpi=150)
    print("\n[INFO] Grafic salvat: results/anomaly_noise_results.png")
    plt.show()


def run_anomaly_scarcity_experiment(full_train_subset, test_loader, config, device,
                                    fractions=None, n_seeds=5):
    """
    Scarcity pe anomaly detection: variaza cantitatea de date Normale de antrenament.
    Ambele modele (PINN-AD si AE) sunt antrenate EXCLUSIV pe date Normale.
    Argumentul pentru paper: PINN-ul mentine AUROC mai bun cu putine date Normale
    datorita inductiei fizice (ODE structureaza spatiul τ chiar si cu 100 ferestre).
    """
    if fractions is None:
        fractions = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    actual_dataset = full_train_subset.dataset
    indices        = np.array(full_train_subset.indices)
    train_labels   = np.array([
        actual_dataset.labels[actual_dataset.valid_indices[idx] + actual_dataset.time_steps - 1]
        for idx in indices
    ])
    class0_idx = indices[train_labels == 0]
    n_normal   = len(class0_idx)

    _pin = torch.cuda.is_available()
    _nw  = min(4, (os.cpu_count() or 1) // 2)

    binary_labels = np.array([
        (actual_dataset.labels[actual_dataset.valid_indices[idx] + actual_dataset.time_steps - 1] != 0)
        for idx in np.array(test_loader.dataset.indices)
    ], dtype=int)

    pinn_aurocs_mean, pinn_aurocs_std = [], []
    ae_aurocs_mean,   ae_aurocs_std   = [], []

    n_total_runs = len(fractions) * n_seeds
    run_idx = 0
    print(f"\n[INFO] {len(fractions)} fractii x {n_seeds} seed-uri = {n_total_runs} runde de antrenament\n")

    for frac_idx, frac in enumerate(fractions):
        n_sub = max(32, int(n_normal * frac))
        pinn_runs, ae_runs = [], []

        print(f"[{frac_idx+1}/{len(fractions)}] Fractie={frac:.2f}  ({n_sub} ferestre normale din {n_normal})")

        for seed in range(n_seeds):
            run_idx += 1
            rng   = np.random.default_rng(seed)
            sub   = rng.choice(class0_idx, size=n_sub, replace=False)
            loader = DataLoader(
                Subset(actual_dataset, sub),
                batch_size=config['batch_size'], shuffle=True,
                num_workers=_nw, pin_memory=_pin, persistent_workers=_nw > 0,
            )

            print(f"  Seed {seed+1}/{n_seeds}  (runda {run_idx}/{n_total_runs})"
                  f"  → PINN-AD...", flush=True)
            pinn = train_pinn_anomaly(loader, config, device)

            print(f"  Seed {seed+1}/{n_seeds}  → Calibrare PINN...", flush=True)
            pinn.calibrate(loader, device, mode='combined')

            print(f"  Seed {seed+1}/{n_seeds}  → Autoencoder...", flush=True)
            ae = train_autoencoder(loader, config, device)

            print(f"  Seed {seed+1}/{n_seeds}  → Evaluare scoruri...", flush=True)
            ps,   _, _, _ = _collect_scores(pinn, test_loader, device, is_pinn=True)
            ae_s, _, _, _ = _collect_scores(ae,   test_loader, device)
            p_auc = roc_auc_score(binary_labels, ps)
            a_auc = roc_auc_score(binary_labels, ae_s)
            pinn_runs.append(p_auc)
            ae_runs.append(a_auc)
            winner = "PINN ✓" if p_auc > a_auc else "AE  ✓"
            print(f"    AUROC: PINN={p_auc:.4f}  AE={a_auc:.4f}  → {winner}")

        pm, ps_ = np.mean(pinn_runs), np.std(pinn_runs)
        am, as_ = np.mean(ae_runs),   np.std(ae_runs)
        pinn_aurocs_mean.append(pm); pinn_aurocs_std.append(ps_)
        ae_aurocs_mean.append(am);   ae_aurocs_std.append(as_)
        winner_avg = "PINN ✓" if pm > am else "AE  ✓"
        print(f"  ── Medie fractie {frac:.2f}: PINN={pm:.4f}±{ps_:.4f}  "
              f"AE={am:.4f}±{as_:.4f}  → {winner_avg}\n")

    _plot_scarcity(fractions, pinn_aurocs_mean, pinn_aurocs_std,
                   ae_aurocs_mean, ae_aurocs_std, n_normal)
    return {
        'fractions':       fractions,
        'pinn_auroc_mean': pinn_aurocs_mean,
        'pinn_auroc_std':  pinn_aurocs_std,
        'ae_auroc_mean':   ae_aurocs_mean,
        'ae_auroc_std':    ae_aurocs_std,
    }


def _plot_scarcity(fractions, pinn_mean, pinn_std, ae_mean, ae_std, n_normal):
    x = np.array([max(32, int(f * n_normal)) for f in fractions])
    pm = np.array(pinn_mean); ps = np.array(pinn_std)
    am = np.array(ae_mean);   as_ = np.array(ae_std)
    delta = pm - am

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle('Scarcity Anomaly Detection: PINN vs Autoencoder\n'
                 'Ambele modele antrenate EXCLUSIV pe date Normale',
                 fontsize=13, fontweight='bold')

    # ── stânga: AUROC vs N ────────────────────────────────────
    ax = axes[0]
    ax.fill_between(x, pm - ps, pm + ps, alpha=0.18, color='#2ca02c')
    ax.fill_between(x, am - as_, am + as_, alpha=0.18, color='#d62728')
    ax.plot(x, pm, 'o-',  color='#2ca02c', lw=2.5, ms=7,
            label='PINN-AD (features ODE)')
    ax.plot(x, am, 's--', color='#d62728', lw=2.5, ms=7,
            label='Autoencoder CNN')

    # adnotare la cel mai mic N
    ax.annotate(f'PINN={pm[-1]:.3f}', xy=(x[-1], pm[-1]),
                xytext=(x[-1]*1.15, pm[-1]+0.005),
                fontsize=8, color='#2ca02c', fontweight='bold')
    ax.annotate(f'AE={am[-1]:.3f}', xy=(x[-1], am[-1]),
                xytext=(x[-1]*1.15, am[-1]-0.012),
                fontsize=8, color='#d62728', fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlabel('Ferestre Normale de antrenament  (scală log)', fontsize=11)
    ax.set_ylabel('AUROC  (Normal vs Orice Defect)', fontsize=11)

    # zoom pe zona de interes
    ymin = max(0.4, min(am) - 0.08)
    ymax = min(1.01, max(pm) + 0.04)
    ax.set_ylim(ymin, ymax)

    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='x', which='both', labelsize=9)

    # ── dreapta: avantaj PINN (ΔAUROC) ───────────────────────
    ax2 = axes[1]
    ax2.fill_between(x, 0, delta, where=delta >= 0,
                     alpha=0.45, color='#2ca02c', label='PINN câștigă')
    ax2.fill_between(x, 0, delta, where=delta < 0,
                     alpha=0.45, color='#d62728', label='AE câștigă')
    ax2.plot(x, delta, 'ko-', markersize=5, lw=1.8)
    ax2.axhline(0, color='black', lw=0.8, linestyle='-')
    ax2.set_xscale('log')
    ax2.set_xlabel('Ferestre Normale de antrenament', fontsize=10)
    ax2.set_ylabel('ΔAUROC  (PINN − AE)', fontsize=10)
    ax2.set_title('Avantaj PINN\n(mai mare = mai bun)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(axis='x', which='both', labelsize=9)

    plt.tight_layout()
    plt.savefig('results/anomaly_scarcity_results.png', dpi=150, bbox_inches='tight')
    print("\n[INFO] Grafic salvat: results/anomaly_scarcity_results.png")
    plt.show()


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
