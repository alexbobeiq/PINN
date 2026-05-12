import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. ÎNCĂRCAREA ȘI CURĂȚAREA DATELOR
# ==========================================
df = pd.read_csv('date_vtem.csv')
coloane_inutile = [
    'Presiune1_Valva2', 'Presiune2_Valva2',
    'Presiune1_Valva3', 'Presiune2_Valva3',
    'Presiune1_Valva4', 'Presiune2_Valva4'
]
df = df.drop(columns=coloane_inutile, errors='ignore')

# ==========================================
# 2. PROCESAREA TIMPULUI ȘI A PRESIUNILOR
# ==========================================
if 'Timestamp' in df.columns:
    timp_start = df['Timestamp'].iloc[0]
    df['Timestamp'] = (df['Timestamp'] - timp_start) / 1000.0
else:
    print("Atenție: Nu s-a găsit coloana 'Timestamp'.")

for col in ['Presiune1_Valva1', 'Presiune2_Valva1']:
    if col in df.columns:
        df.loc[df[col] > 32767, col] = df[col] - 65536

# ==========================================
# 3. CONFIGURARE NUME ȘI CULORI
# ==========================================
nume_profesioniste = {
    'Presiune1_Valva1': 'P1 (Cameră Activă) [mbar]',
    'Presiune2_Valva1': 'P2 (Evacuare) [mbar]',
    'pre_input': 'Setpoint Alimentare [mbar]',
    'ex_0': 'Grad Deschidere Evacuare'
}

mapare_clase = {
    0: ('Normal (0)', '#2ca02c'),      # Verde
    1: ('Defect F1 (1)', '#1f77b4'),   # Albastru
    2: ('Defect F2 (2)', '#ff7f0e'),   # Portocaliu
    3: ('Defect F3 (3)', '#d62728'),   # Roșu
    4: ('Defect F4 (4)', '#9467bd'),   # Mov
    5: ('Defect F5 (5)', '#8c564b'),   # Maro
    6: ('Defect F6 (6)', '#e377c2'),   # Roz
    7: ('Defect F7 (7)', '#7f7f7f')    # Gri
}

# ==========================================
# 4. CONSTRUIREA GRAFICULUI (MATPLOTLIB)
# ==========================================
coloane_de_plotat = list(nume_profesioniste.keys())
clase_gasite = sorted(df['Label'].unique()) if 'Label' in df.columns else []

# Creăm figura cu 4 subgrafice (sharex=True aliniază automat axa X)
fig, axs = plt.subplots(len(coloane_de_plotat), 1, figsize=(14, 10), sharex=True)

# Dicționar pentru a păstra elementele legendei (să nu fie duplicată de 4 ori)
legend_handles = {}

for i, col in enumerate(coloane_de_plotat):
    if col not in df.columns:
        continue
        
    ax = axs[i]
    axa_x = df['Timestamp'] if 'Timestamp' in df.columns else df.index
    
    # 1. Linia gri de fundal (zorder=1 o ține în spate)
    ax.plot(axa_x, df[col], color='lightgrey', linewidth=1, zorder=1)
    
    # 2. Punctele colorate (zorder=2 le pune deasupra liniei)
    for cls in clase_gasite:
        if cls in mapare_clase:
            nume_eticheta, culoare = mapare_clase[cls]
            mask = df['Label'] == cls
            axa_x_mascat = df.loc[mask, 'Timestamp'] if 'Timestamp' in df.columns else df.loc[mask].index
            
            scatter = ax.scatter(
                axa_x_mascat, 
                df.loc[mask, col], 
                color=culoare, 
                s=8, # Dimensiunea punctului
                zorder=2,
                label=nume_eticheta
            )
            
            # Salvăm o singură instanță în legendă pentru fiecare clasă detectată
            if nume_eticheta not in legend_handles:
                legend_handles[nume_eticheta] = scatter
                
    # Titlul fiecărui subplot și grid-ul
    ax.set_title(nume_profesioniste[col], fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Design curat (fără margini sus și dreapta)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ==========================================
# 5. ASPECT ȘI AFIȘARE
# ==========================================
axs[-1].set_xlabel("Timp Relativ (Secunde)", fontsize=11)
fig.suptitle("Dinamica Sistemului Pneumatic - Diagnoză", fontsize=14, fontweight='bold', y=0.98)

# Adăugăm legenda globală în dreapta
fig.legend(
    legend_handles.values(), 
    legend_handles.keys(), 
    title="Stare / Defect", 
    loc='center right', 
    bbox_to_anchor=(1.12, 0.5) # O împingem puțin în afara graficelor
)

# Ajustăm marginile ca să încapă totul perfect și fără suprapuneri
plt.tight_layout()
fig.subplots_adjust(right=0.88, top=0.92, hspace=0.3)

plt.show()
