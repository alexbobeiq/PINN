"""
Script de achizitie date de la PLC prin Modbus.
Suporta rulari multiple cu append la CSV-ul existent.

Utilizare:
    python achizitie_date.py             # append la date_vtem.csv
    python achizitie_date.py --new       # fisier nou cu timestamp
    python achizitie_date.py --check     # afiseaza statistici dataset curent

Registre Modbus (pHolding):
    [0]  Presiune1_Valva1  (offset +500 la clasa 7 manual)
    [1]  Presiune1_Valva2
    [2]  Presiune2_Valva1
    [3]  Presiune2_Valva2
    [4]  Presiune1_Valva3
    [5]  Presiune2_Valva3
    [6]  Presiune1_Valva4
    [7]  Presiune2_Valva4
    [8]  pre_input
    [9]  ex_0
    [10] H
    [11] Protocol_Label  (65535 = -1 = tranzitie)
    [12] protocol_step   (0-56; 57 = protocol complet)
"""

import csv
import time
import ctypes
import argparse
import os
from datetime import datetime

try:
    import minimalmodbus
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False
    print("[WARN] minimalmodbus nu e instalat. Ruleaza: pip install minimalmodbus")

# ─── Configurare ───────────────────────────────────────────────
MODBUS_PORT    = 'COM3'       # portul serial al PLC-ului
MODBUS_SLAVE   = 1
MODBUS_BAUD    = 19200
SAMPLE_HZ      = 10           # frecventa achizitie (Hz)
CSV_PATH       = 'date_vtem.csv'
N_REGISTERS    = 13           # pHolding[0..12]
DONE_STEP      = 57           # protocol_step la care protocolul s-a terminat

COLUMNS = [
    'Timestamp',
    'Presiune1_Valva1', 'Presiune2_Valva1',
    'Presiune1_Valva2', 'Presiune2_Valva2',
    'Presiune1_Valva3', 'Presiune2_Valva3',
    'Presiune1_Valva4', 'Presiune2_Valva4',
    'pre_input', 'ex_0', 'H',
    'Label', 'protocol_step'
]
# ───────────────────────────────────────────────────────────────


def decode_label(raw):
    """65535 (0xFFFF) = -1 = tranzitie; 0-7 = stari stabile."""
    return ctypes.c_int16(raw & 0xFFFF).value


def check_dataset(path):
    """Afiseaza statistici despre dataset-ul curent."""
    import pandas as pd
    if not os.path.exists(path):
        print(f"Fisierul {path} nu exista inca.")
        return

    df = pd.read_csv(path)
    dt = df['Timestamp'].diff().dropna().mean()
    hz = 1000 / dt if dt > 0 else 0

    print(f"\nDataset: {path}")
    print(f"Total esantioane : {len(df):,}")
    print(f"Frecventa medie  : {hz:.1f} Hz")
    print(f"Durata totala    : {len(df)/hz/60:.1f} min\n")

    counts = df['Label'].value_counts().sort_index()
    print(f"{'Label':<8} {'Esantioane':<14} {'Durata':>8}  {'Train(80%)':>12}  {'La 1%':>8}")
    print("─" * 58)
    for label, count in counts.items():
        dur   = count / hz
        train = int(count * 0.8)
        at1pct = max(1, int(train * 0.01))
        flag = " ←  putin!" if count < 1500 else ""
        print(f"  {label:<6} {count:<14,} {dur:>7.0f}s  {train:>12,}  {at1pct:>8}{flag}")


def acquire(csv_path, new_file=False):
    if not MODBUS_AVAILABLE:
        print("Instaleaza minimalmodbus inainte de a rula achizitia.")
        return

    instrument = minimalmodbus.Instrument(MODBUS_PORT, MODBUS_SLAVE)
    instrument.serial.baudrate = MODBUS_BAUD
    instrument.serial.timeout  = 0.5

    # Daca fisierul exista si nu se cere fisier nou → append fara header
    file_exists = os.path.exists(csv_path) and not new_file
    mode = 'a' if file_exists else 'w'

    if new_file:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f'date_vtem_{ts}.csv'
        print(f"[INFO] Fisier nou: {csv_path}")
    elif file_exists:
        print(f"[INFO] Append la fisierul existent: {csv_path}")
    else:
        print(f"[INFO] Fisier nou: {csv_path}")

    interval = 1.0 / SAMPLE_HZ
    sample_count = 0
    transition_count = 0

    print("[INFO] Astept pornirea protocolului PLC (protocol_active = TRUE)...")
    print("[INFO] Ctrl+C pentru oprire manuala.\n")

    with open(csv_path, mode, newline='') as f:
        writer = csv.writer(f)
        if not file_exists or new_file:
            writer.writerow(COLUMNS)

        try:
            while True:
                t_start = time.time()

                regs = instrument.read_registers(0, N_REGISTERS, functioncode=3)

                label         = decode_label(regs[11])
                protocol_step = regs[12]
                timestamp_ms  = int(time.time() * 1000)

                row = [
                    timestamp_ms,
                    regs[0], regs[2],   # P1_V1, P2_V1
                    regs[1], regs[3],   # P1_V2, P2_V2
                    regs[4], regs[5],   # P1_V3, P2_V3
                    regs[6], regs[7],   # P1_V4, P2_V4
                    regs[8], regs[9], regs[10],  # pre_input, ex_0, H
                    label,
                    protocol_step
                ]
                writer.writerow(row)
                sample_count += 1

                if label == -1:
                    transition_count += 1

                # Status la fiecare 100 esantioane
                if sample_count % 100 == 0:
                    pct_trans = 100 * transition_count / sample_count
                    print(f"  [t={sample_count/SAMPLE_HZ:.0f}s] "
                          f"step={protocol_step:2d}  label={label:3d}  "
                          f"P1={regs[0]:5d}  P2={regs[2]:5d}  "
                          f"tranzitii={pct_trans:.0f}%")

                # Oprire automata la finalul protocolului
                if protocol_step >= DONE_STEP:
                    print(f"\n[OK] Protocol complet (step={DONE_STEP}).")
                    break

                # Pauza pentru a mentine frecventa de esantionare
                elapsed = time.time() - t_start
                wait    = max(0, interval - elapsed)
                time.sleep(wait)

        except KeyboardInterrupt:
            print("\n[STOP] Achizitie oprita manual.")

    print(f"\nSalvat {sample_count} esantioane in {csv_path}")
    print(f"Tranzitii (label=-1): {transition_count} ({100*transition_count/max(1,sample_count):.0f}%)")
    check_dataset(csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Achizitie date VTEM prin Modbus')
    parser.add_argument('--new',   action='store_true', help='Creeaza fisier nou cu timestamp')
    parser.add_argument('--check', action='store_true', help='Afiseaza statistici dataset curent')
    parser.add_argument('--port',  default=MODBUS_PORT, help=f'Port serial (default: {MODBUS_PORT})')
    parser.add_argument('--hz',    default=SAMPLE_HZ, type=float, help=f'Frecventa Hz (default: {SAMPLE_HZ})')
    parser.add_argument('--csv',   default=CSV_PATH, help=f'Fisier CSV (default: {CSV_PATH})')
    args = parser.parse_args()

    MODBUS_PORT = args.port
    SAMPLE_HZ   = args.hz
    CSV_PATH    = args.csv

    if args.check:
        check_dataset(args.csv)
    else:
        acquire(args.csv, new_file=args.new)
