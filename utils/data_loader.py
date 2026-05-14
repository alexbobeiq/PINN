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
    
    # --- SPLIT TEMPORAL PE CLASE (Solutia Suprema) ---
    # Elimina Data Leakage-ul si garanteaza prezenta tuturor claselor in Test
    train_indices = []
    test_indices = []
    
    # Etichetele pe care le returneaza dataset-ul sunt decalate cu time_steps
    valid_labels = labels[time_steps:]
    unique_classes = np.unique(valid_labels)
    
    for c in unique_classes:
        c_indices = np.where(valid_labels == c)[0]
        split_point = int(0.8 * len(c_indices))
        
        train_idxs = c_indices[:split_point]
        test_idxs = c_indices[split_point:]
        
        # PURGE GAP: Asiguram ca testul nu se suprapune deloc cu antrenarea.
        # Sarim peste indecsii de test care s-ar suprapune cu fereastra de 60 de pasi a ultimului train.
        if len(train_idxs) > 0 and len(test_idxs) > 0:
            last_train_idx = train_idxs[-1]
            test_idxs = test_idxs[test_idxs >= last_train_idx + time_steps]
            
        train_indices.extend(train_idxs)
        test_indices.extend(test_idxs)
        
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
            scaler)