import pandas as pd
df = pd.read_csv('../data/raw/date_vtem.csv')
print('Shape:', df.shape)
print('\nLabel distribution:')
print(df['Label'].value_counts().sort_index())
print('\nColumns:', list(df.columns))
print('\npre_input unique:', df['pre_input'].unique())
print('\nex_0 unique:', df['ex_0'].unique())
print('\nPresiune1 stats:')
print(df['Presiune1_Valva1'].describe())
print('\nPresiune2 stats:')
print(df['Presiune2_Valva1'].describe())

# Check per-label pressure stats
for label in sorted(df['Label'].unique()):
    subset = df[df['Label'] == label]
    p1 = subset['Presiune1_Valva1'].copy()
    p1[p1 > 32767] = p1 - 65536
    p2 = subset['Presiune2_Valva1'].copy()
    p2[p2 > 32767] = p2 - 65536
    print(f"\nLabel {label}: count={len(subset)}, P1 mean={p1.mean():.1f}, P2 mean={p2.mean():.1f}, pre_input={subset['pre_input'].unique()}, ex_0={subset['ex_0'].unique()}")
