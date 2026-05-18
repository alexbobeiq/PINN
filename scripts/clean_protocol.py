"""
Curatare date_protocol.csv
- Coloane pastrate: Timestamp, Presiune1_Valva1, Presiune2_Valva1, pre_input, ex_0, Label
- Underflow uint16 -> int16 semnat (65534 -> -2, 65535 -> -1, etc.)
- Gap-uri >1s: elimina 5 randuri inainte+dupa, adauga coloana 'segment'
- Nota: esantionarea reala este ~20ms (NU 10ms cum era asteptat)
"""

import os
import numpy as np
import pandas as pd

_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT  = os.path.join(_DIR, "../data/raw/date_protocol.csv")
OUTPUT = os.path.join(_DIR, "../data/processed/date_protocol_clean.csv")

GAP_THRESHOLD_MS = 1000   # gap considerat discontinuitate
EDGE_ROWS = 5             # randuri eliminate la capetele fiecarui gap

COLS_KEEP = ["Timestamp", "Presiune1_Valva1", "Presiune2_Valva1", "pre_input", "ex_0", "Label"]
COLS_SIGNED = ["Presiune1_Valva1", "Presiune2_Valva1"]  # coloane cu potential underflow


def to_signed16(series: pd.Series) -> pd.Series:
    vals = series.astype(np.int64)
    return np.where(vals >= 32768, vals - 65536, vals).astype(np.int32)


def main():
    print("Citire fisier...")
    df = pd.read_csv(INPUT)

    # Fix trailing space in column names
    df.columns = [c.strip() for c in df.columns]

    print(f"Rows initiale: {len(df)}")
    print(f"Coloane disponibile: {list(df.columns)}")

    # Pastreaza doar coloanele de interes
    df = df[COLS_KEEP].copy()

    # Conversie uint16 -> int16 semnat pentru coloanele cu underflow
    for col in COLS_SIGNED:
        n_uf = (df[col] >= 32768).sum()
        df[col] = to_signed16(df[col])
        print(f"  {col}: {n_uf} valori convertite la int16 semnat "
              f"(min={df[col].min()}, max={df[col].max()})")

    # Detectare gap-uri mari
    df["Timestamp"] = df["Timestamp"].astype(np.int64)
    diffs = df["Timestamp"].diff().fillna(0)
    gap_indices = diffs[diffs > GAP_THRESHOLD_MS].index.tolist()

    print(f"\nGap-uri detectate (>{GAP_THRESHOLD_MS}ms): {len(gap_indices)}")
    for idx in gap_indices:
        pos = df.index.get_loc(idx)
        gap_ms = int(diffs[idx])
        print(f"  Row {pos+1}: gap={gap_ms}ms ({gap_ms/1000:.1f}s)")

    # Marcare randuri de eliminat (EDGE_ROWS inainte si dupa fiecare gap)
    rows_to_drop = set()
    for idx in gap_indices:
        pos = df.index.get_loc(idx)
        # EDGE_ROWS dupa gap (inclusiv primul rand dupa)
        for offset in range(EDGE_ROWS):
            if pos + offset < len(df):
                rows_to_drop.add(df.index[pos + offset])
        # EDGE_ROWS inainte de gap
        for offset in range(1, EDGE_ROWS + 1):
            if pos - offset >= 0:
                rows_to_drop.add(df.index[pos - offset])

    print(f"Randuri eliminate la capetele gap-urilor: {len(rows_to_drop)}")
    df = df.drop(index=list(rows_to_drop)).reset_index(drop=True)

    # Clasa 7 = eroare senzor: +500 nu era aplicat in modul protocol → aplicam acum
    n7 = (df["Label"] == 7).sum()
    df.loc[df["Label"] == 7, "Presiune1_Valva1"] += 500
    print(f"\nClasa 7: +500 aplicat la Presiune1_Valva1 ({n7} randuri)")

    # Adauga coloana 'segment' - ID segment continuu intre gap-uri
    diffs2 = df["Timestamp"].diff().fillna(0)
    new_gaps = diffs2[diffs2 > GAP_THRESHOLD_MS].index.tolist()
    segment_id = 0
    segment_col = np.zeros(len(df), dtype=np.int32)
    for i in range(len(df)):
        if i in new_gaps:
            segment_id += 1
        segment_col[i] = segment_id
    df["segment"] = segment_col

    print(f"\nSegmente detectate: {segment_id + 1}")

    # Statistici finale
    print(f"\nRows dupa curatare: {len(df)}")
    print(f"Rows eliminate total: {308013 - len(df)}")  # original count

    # Verificare esantionare
    diffs_final = df["Timestamp"].groupby(df["segment"]).apply(
        lambda g: g.diff().dropna()
    )
    all_diffs = diffs_final.values
    print(f"\nEsantionare in datele curatate:")
    print(f"  Media: {np.mean(all_diffs):.2f}ms")
    print(f"  Mediana: {np.median(all_diffs):.2f}ms")
    print(f"  Min: {np.min(all_diffs):.0f}ms, Max: {np.max(all_diffs):.0f}ms")
    print(f"  NOTE: esantionarea reala este ~20ms, NU 10ms")

    # Distributie Label
    print(f"\nDistributie Label:")
    print(df["Label"].value_counts().sort_index().to_string())

    # Salvare
    df.to_csv(OUTPUT, index=False)
    print(f"\nFisier salvat: {OUTPUT}")


if __name__ == "__main__":
    main()
