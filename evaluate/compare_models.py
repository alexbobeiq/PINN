import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from train.train_data_driven import train_cnn
from train.train_pinn import train_pinn

def get_f1_metrics(model, loader, device, is_pinn=False):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            x_scaled, setpoints, y = batch
            x_scaled = x_scaled.to(device)
            
            if is_pinn:
                logits, _ = model(x_scaled)
            else:
                logits = model(x_scaled)
            
            pred = torch.argmax(logits, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(pred.cpu().numpy())
    return f1_score(y_true, y_pred, average='weighted')

def run_scarcity_experiment(full_train_subset, test_loader, config_cnn, config_pinn, device,
                            fractions=[1.0, 0.8, 0.6, 0.4, 0.3, 0.1, 0.05], n_seeds=3):
    results = {'cnn': [], 'pinn': []}

    actual_dataset = full_train_subset.dataset
    indices = full_train_subset.indices
    train_labels = np.array([actual_dataset.labels[idx + actual_dataset.time_steps] for idx in indices])

    for frac in fractions:
        print(f"\n{'='*50}")
        print(f"  TEST: {frac*100:.0f}% DIN DATE ({int(len(indices)*frac)} samples)")
        print(f"{'='*50}")

        cnn_scores = []
        pinn_scores = []
        
        for seed in range(n_seeds):
            print(f"\n  --- Seed {seed+1}/{n_seeds} ---")
            
            if frac < 1.0:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=42 + seed)
                local_idx, _ = next(sss.split(np.zeros(len(train_labels)), train_labels))
                final_indices = [indices[i] for i in local_idx]
                train_subset = Subset(actual_dataset, final_indices)
            else:
                train_subset = full_train_subset

            scarcity_loader = DataLoader(train_subset, batch_size=config_cnn['batch_size'], shuffle=True)

            # Train and evaluate CNN
            print(f"  >> Antrenare CNN ({config_cnn['epochs']} epoci)...")
            cnn_model = train_cnn(scarcity_loader, config_cnn, device)
            f1_cnn = get_f1_metrics(cnn_model, test_loader, device)
            cnn_scores.append(f1_cnn)
            print(f"  >> CNN F1 = {f1_cnn:.4f}")

            # Train and evaluate PINN
            print(f"  >> Antrenare PINN ({config_pinn['epochs']} epoci)...")
            pinn_model = train_pinn(scarcity_loader, config_pinn, device)
            f1_pinn = get_f1_metrics(pinn_model, test_loader, device, is_pinn=True)
            pinn_scores.append(f1_pinn)
            print(f"  >> PINN F1 = {f1_pinn:.4f}")
        
        mean_cnn = np.mean(cnn_scores)
        mean_pinn = np.mean(pinn_scores)
        results['cnn'].append(mean_cnn)
        results['pinn'].append(mean_pinn)
        winner = "PINN" if mean_pinn > mean_cnn else "CNN"
        print(f"\n  >> REZULTAT {frac*100:.0f}%: CNN={mean_cnn:.4f}, PINN={mean_pinn:.4f} -> Castigator: {winner}")

    plot_results(fractions, results)
    return results

def plot_results(fractions, results):
    plt.figure(figsize=(10, 6))
    x_vals = [f * 100 for f in fractions]
    plt.plot(x_vals, results['cnn'], 'o--', label='CNN (Statistic)', color='red', markersize=8)
    plt.plot(x_vals, results['pinn'], 's-', label='PINN (Fizica)', color='green', markersize=8)
    plt.xlabel('% Date de antrenament', fontsize=12)
    plt.ylabel('Scor F1 Weighted', fontsize=12)
    plt.title('Rezultatul Experimentului: PINN vs CNN (Data Scarcity)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig('results_comparison.png', dpi=150)
    plt.show()