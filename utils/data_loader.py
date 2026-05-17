import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

PRESSURE_COLS = ['Presiune1_Valva1', 'Presiune2_Valva1']
SETPOINT_COLS = ['pre_input', 'ex_0']


class PneumaticDataset(Dataset):
    def __init__(self, features_scaled, sp_alim, ex_0_arr, labels, valid_indices, time_steps):
        self.features_scaled = features_scaled
        self.sp_alim = sp_alim
        self.ex_0 = ex_0_arr
        self.labels = labels
        self.valid_indices = valid_indices
        self.time_steps = time_steps

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        x = self.features_scaled[i : i + self.time_steps]
        last = i + self.time_steps - 1
        sp = self.sp_alim[last]
        ex = self.ex_0[last]
        y = self.labels[last]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor([sp, ex], dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )


def _fix_underflow(df):
    for col in PRESSURE_COLS:
        if col in df.columns:
            df.loc[df[col] > 32767, col] = df[col] - 65536
    return df


def _build_valid_indices(segments, time_steps):
    """Returneaza indicii de start ai ferestrelor care raman in acelasi segment."""
    seg = np.array(segments)
    # O fereastra [i, i+T) e valida daca seg[i] == seg[i+T-1]
    starts = np.arange(len(seg) - time_steps)
    mask = seg[starts] == seg[starts + time_steps - 1]
    return starts[mask]


def _segment_split(segments, labels, valid_indices, time_steps, test_fraction=0.2):
    """
    Leave-segment-out split: pentru fiecare clasa, ultimele ceil(n_seg * test_fraction)
    segmente merg la test, restul la train.
    Returneaza (train_indices, test_indices) din valid_indices.
    """
    seg = np.array(segments)
    lbl = np.array(labels)

    test_segments = set()
    for cls in np.unique(lbl):
        cls_segs = np.unique(seg[lbl == cls])
        n_test = max(1, int(np.ceil(len(cls_segs) * test_fraction)))
        test_segments.update(cls_segs[-n_test:].tolist())

    # Fiecare fereastra e clasificata dupa segmentul ultimului sample
    window_segs = seg[valid_indices + time_steps - 1]
    train_mask = ~np.isin(window_segs, list(test_segments))
    test_mask = np.isin(window_segs, list(test_segments))

    return valid_indices[train_mask], valid_indices[test_mask]


def _temporal_split(labels, valid_indices, time_steps, test_fraction=0.2):
    """Split temporal 80/20 per clasa (comportament original pentru date fara segment)."""
    lbl = np.array(labels)
    window_labels = lbl[valid_indices + time_steps - 1]

    train_indices, test_indices = [], []
    for cls in np.unique(window_labels):
        c_idx = valid_indices[window_labels == cls]
        split = int((1 - test_fraction) * len(c_idx))
        train_indices.append(c_idx[:split])
        if len(c_idx[split:]) > 0:
            # Purge gap: evita suprapunerea ferestrelor
            gap_start = c_idx[split - 1] + time_steps if split > 0 else 0
            safe = c_idx[split:][c_idx[split:] >= gap_start]
            test_indices.append(safe)

    return np.concatenate(train_indices), np.concatenate(test_indices)


def get_dataloaders(data_path, time_steps, batch_size):
    """
    Incarca datele si returneaza train/test DataLoader + scaler.

    Daca CSV-ul contine coloana 'segment' (date_protocol_clean):
        → leave-segment-out split (20% segmente per clasa la test)
    Altfel (date_vtem):
        → split temporal 80/20 per clasa
    """
    df = pd.read_csv(data_path)
    df = _fix_underflow(df)

    raw = df[PRESSURE_COLS].values.astype(np.float32)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw).astype(np.float32)

    if 'pre_input' in df.columns:
        sp_alim = df['pre_input'].values.astype(np.float32)
        ex_0_arr = df['ex_0'].values.astype(np.float32)
    else:
        sp_alim = np.full(len(df), 2500.0, dtype=np.float32)
        ex_0_arr = np.full(len(df), 10000.0, dtype=np.float32)

    labels = df['Label'].values

    has_segments = 'segment' in df.columns
    segments = df['segment'].values if has_segments else np.zeros(len(df), dtype=int)

    valid_indices = _build_valid_indices(segments, time_steps)

    if has_segments:
        train_idx, test_idx = _segment_split(segments, labels, valid_indices, time_steps)
        split_type = "leave-segment-out"
    else:
        train_idx, test_idx = _temporal_split(labels, valid_indices, time_steps)
        split_type = "temporal 80/20"

    print(f"[DataLoader] Split: {split_type}")
    print(f"[DataLoader] Ferestre valide: {len(valid_indices)} "
          f"(train={len(train_idx)}, test={len(test_idx)})")

    dataset = PneumaticDataset(scaled, sp_alim, ex_0_arr, labels, valid_indices, time_steps)
    train_ds = torch.utils.data.Subset(dataset, np.where(np.isin(valid_indices, train_idx))[0])
    test_ds  = torch.utils.data.Subset(dataset, np.where(np.isin(valid_indices, test_idx))[0])

    pin = torch.cuda.is_available()
    nw  = min(4, (os.cpu_count() or 1) // 2)
    pw  = nw > 0
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=nw, pin_memory=pin, persistent_workers=pw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                   num_workers=nw, pin_memory=pin, persistent_workers=pw),
        scaler,
    )


def get_external_test_loader(data_path, time_steps, batch_size, scaler,
                             source_dt_ms=60, target_dt_ms=20):
    """
    Incarca un dataset extern (ex: date_vtem) si il resampleaza la target_dt_ms
    folosind scalarul antrenat pe datele de protocol. Returneaza DataLoader.

    source_dt_ms: rata de esantionare a sursei (ms)
    target_dt_ms: rata la care s-a antrenat modelul (ms)
    """
    df = pd.read_csv(data_path)
    df = _fix_underflow(df)
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # Resampleaza la target_dt_ms prin interpolare liniara
    ratio = source_dt_ms / target_dt_ms
    if abs(ratio - 1.0) > 0.05:
        t_orig = df['Timestamp'].values.astype(np.float64)
        t_new  = np.arange(t_orig[0], t_orig[-1], target_dt_ms)
        resampled = {'Timestamp': t_new}
        for col in PRESSURE_COLS + SETPOINT_COLS:
            if col in df.columns:
                resampled[col] = np.interp(t_new, t_orig, df[col].values.astype(np.float64))
        if 'Label' in df.columns:
            resampled['Label'] = np.round(
                np.interp(t_new, t_orig, df['Label'].values.astype(np.float64))
            ).astype(int)
        df = pd.DataFrame(resampled)
        print(f"[ExternalLoader] Resamplat {source_dt_ms}ms → {target_dt_ms}ms: "
              f"{len(df)} esantioane")

    raw = df[PRESSURE_COLS].values.astype(np.float32)
    scaled = scaler.transform(raw).astype(np.float32)

    if 'pre_input' in df.columns:
        sp_alim = df['pre_input'].values.astype(np.float32)
        ex_0_arr = df['ex_0'].values.astype(np.float32)
    else:
        sp_alim = np.full(len(df), 2500.0, dtype=np.float32)
        ex_0_arr = np.full(len(df), 10000.0, dtype=np.float32)

    labels = df['Label'].values

    # Fara segment — ferestre secventiale simple
    segments = np.zeros(len(df), dtype=int)
    valid_indices = _build_valid_indices(segments, time_steps)

    print(f"[ExternalLoader] Ferestre valide: {len(valid_indices)}")

    dataset = PneumaticDataset(scaled, sp_alim, ex_0_arr, labels, valid_indices, time_steps)
    pin = torch.cuda.is_available()
    nw  = min(4, (os.cpu_count() or 1) // 2)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
