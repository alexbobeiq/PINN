import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class PneumaticDataset(Dataset):
    def __init__(self, features, labels, time_steps):
        self.features = features
        self.labels = labels
        self.time_steps = time_steps

    def __len__(self):
        return len(self.features) - self.time_steps

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.time_steps]
        y = self.labels[idx + self.time_steps]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)  # type: ignore
def get_dataloaders(data_path, time_steps, batch_size):
    df = pd.read_csv(data_path)
    
    # 1. Ștergem coloanele inutile
    coloane_inutile = ['Presiune1_Valva2', 'Presiune2_Valva2', 'Presiune1_Valva3', 'Presiune2_Valva3', 'Presiune1_Valva4', 'Presiune2_Valva4']
    df = df.drop(columns=coloane_inutile, errors='ignore')

    # 2. Corectare Underflow (65535 devine -1 etc.)
    for col in ['Presiune1_Valva1', 'Presiune2_Valva1']:
        if col in df.columns:
            df.loc[df[col] > 32767, col] = df[col] - 65536
            
    # 3. Extragerea caracteristicilor
    feature_cols = ['Presiune1_Valva1', 'Presiune2_Valva1']
    features = df[feature_cols].values
    labels = df['Label'].values
    
    # 4. Normalizare
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 5. Crearea seturilor de date
    dataset = PneumaticDataset(features_scaled, labels, time_steps)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler
