import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class PneumaticDataset(Dataset):
    def __init__(self, features_scaled, sp_alim, ex_0, labels, time_steps):
        self.features_scaled = features_scaled
        self.sp_alim = sp_alim
        self.ex_0 = ex_0
        self.labels = labels
        self.time_steps = time_steps

    def __len__(self):
        return len(self.features_scaled) - self.time_steps

    def __getitem__(self, idx):
        x_scaled = self.features_scaled[idx : idx + self.time_steps]
        
        # Setpointurile fizice de la finalul ferestrei de timp
        # Acestea vor fi folosite DOAR in functia Loss, NU ca input in retea!
        sp = self.sp_alim[idx + self.time_steps]
        ex = self.ex_0[idx + self.time_steps]
        
        y = self.labels[idx + self.time_steps]
        
        return (
            torch.tensor(x_scaled, dtype=torch.float32),
            torch.tensor([sp, ex], dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )

def get_dataloaders(data_path, time_steps, batch_size):
    df = pd.read_csv(data_path)
    
    # 1. Corectare Underflow
    for col in ['Presiune1_Valva1', 'Presiune2_Valva1']:
        if col in df.columns:
            df.loc[df[col] > 32767, col] = df[col] - 65536
            
    # 2. Scalam DOAR presiunile P1 si P2 (Inputul retelei)
    raw_pressures = df[['Presiune1_Valva1', 'Presiune2_Valva1']].values.astype(np.float32)
    scaler = StandardScaler()
    scaled_pressures = scaler.fit_transform(raw_pressures).astype(np.float32)
    
    # 3. Extragem parametrii fizici REALI din dataset (pentru Loss)
    # Bug fix: Inainte erau hardcodati la 2500 si 10000, deci loss-ul nu invata nimic
    if 'pre_input' in df.columns and 'ex_0' in df.columns:
        sp_alim = df['pre_input'].values.astype(np.float32)
        ex_0 = df['ex_0'].values.astype(np.float32)
    else:
        sp_alim = np.full(len(df), 2500.0, dtype=np.float32)
        ex_0 = np.full(len(df), 10000.0, dtype=np.float32)
        
    labels = df['Label'].values
    
    dataset = PneumaticDataset(scaled_pressures, sp_alim, ex_0, labels, time_steps)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
            scaler)