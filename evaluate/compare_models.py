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
        for x, y in loader:
            x = x.to(device)
            if is_pinn:
                logits, _ = model(x)
            else:
                logits = model(x)
            pred = torch.argmax(logits, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(pred.cpu().numpy())
    return f1_score(y_true, y_pred, average='weighted')

def run_scarcity_experiment(full_train_subset, test_loader, config_cnn, config_pinn, device, fractions=[1.0, 0.3, 0.1, 0.05]):
    results = {'cnn': [], 'pinn': []}

    actual_dataset = full_train_subset.dataset
    indices = full_train_subset.indices
    train_labels = np.array([actual_dataset.labels[idx + actual_dataset.time_steps] for idx in indices])

    for frac in fractions:
        print(f"\n--- TEST: {frac*100:.0f}% DIN DATE ---")

        if frac < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=42)
            local_idx, _ = next(sss.split(np.zeros(len(train_labels)), train_labels))
            final_indices = [indices[i] for i in local_idx]
            train_subset = Subset(actual_dataset, final_indices)
        else:
            train_subset = full_train_subset

        scarcity_loader = DataLoader(train_subset, batch_size=config_cnn['batch_size'], shuffle=True)

        # Pasăm device-ul către antrenament
        cnn_model = train_cnn(scarcity_loader, config_cnn, device)
        f1_cnn = get_f1_metrics(cnn_model, test_loader, device)
        results['cnn'].append(f1_cnn)

        pinn_model = train_pinn(scarcity_loader, config_pinn, device)
        f1_pinn = get_f1_metrics(pinn_model, test_loader, device, is_pinn=True)
        results['pinn'].append(f1_pinn)

    plot_results(fractions, results)

def plot_results(fractions, results):
    plt.figure(figsize=(10, 6))
    x_vals = [f * 100 for f in fractions]
    plt.plot(x_vals, results['cnn'], 'o--', label='CNN (Statistic)', color='red')
    plt.plot(x_vals, results['pinn'], 's-', label='PINN (Fizică)', color='green')
    plt.xlabel('% Date de antrenament')
    plt.ylabel('Scor F1 Weighted')
    plt.title('Rezultatul Experimentului: PINN vs CNN (Data Scarcity)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis() 
    plt.show()