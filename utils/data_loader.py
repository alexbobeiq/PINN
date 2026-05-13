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
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def get_dataloaders(data_path, time_steps, batch_size):
    df = pd.read_csv(data_path)
    
    # 1. Corectare Underflow
    for col in ['Presiune1_Valva1', 'Presiune2_Valva1']:
        if col in df.columns:
            df.loc[df[col] > 32767, col] = df[col] - 65536
            
    # 2. Scalăm DOAR presiunile reale (P1, P2)
    scaler = StandardScaler()
    pressures_scaled = scaler.fit_transform(df[['Presiune1_Valva1', 'Presiune2_Valva1']])
    
    # 3. Forțăm setpoint-urile NOMINALE (2500 mbar și 10000 deschidere)
    # Acestea vor fi coloanele 2 și 3 din tensorul final
    sp_input = np.full((len(df), 1), 2500)
    sp_exhaust = np.full((len(df), 1), 10000)
    
    # Concatenăm totul într-o matrice de [N x 4]
    features = np.hstack((pressures_scaled, sp_input, sp_exhaust))
    labels = df['Label'].values
    
    dataset = PneumaticDataset(features, labels, time_steps)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=False), scaler