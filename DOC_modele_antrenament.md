# Documentație Modele și Antrenament

## Arhitectura generală a proiectului

Proiectul compară două abordări pentru detecția anomaliilor în sistemul pneumatic VTEM:

| | PINN-AD | Autoencoder CNN |
|---|---|---|
| **Tip** | Physics-Informed Neural Network | Rețea neurală convoluțională |
| **Antrenament** | Minimizează reziduul ecuației ODE | Minimizează eroarea de reconstrucție |
| **Scor anomalie** | Distanță Mahalanobis în spațiu de features | MSE de reconstrucție |
| **Prior** | Ecuația ODE pneumatică τ·dP/dt + P = P_eq | Niciunul (pur date) |

Ambele modele au **aceeași arhitectură CNN backbone** — diferă doar funcția de pierdere și modul de scorare a anomaliei.

---

## 1. Datele de intrare

### Formatul unei ferestre
Fiecare exemplu este o fereastră temporală de **T=60 de pași** cu **2 canale**:
```
x ∈ ℝ^{60 × 2}

x[:, 0] = Presiune1_Valva1  (normalizată: medie 0, std 1)
x[:, 1] = Presiune2_Valva1  (normalizată: medie 0, std 1)
```

### Semnificație fizică
- **Presiune1**: camera de alimentare a actuatorului — crește în faza de avansare (FILL)
- **Presiune2**: camera de evacuare — scade în faza de retragere (EXHAUST)

Rata de eșantionare reală este ~20ms → T=60 acoperă ~1.2 secunde, adică aproximativ **1 ciclu complet** de avansare + retragere.

---

## 2. PINN-AD — Physics-Informed Anomaly Detector

**Fișier:** `models/pinn_anomaly.py`

### 2.1 Backbone CNN (Encoder)

```
Input: x ∈ ℝ^{60 × 2}   (transpus la ℝ^{2 × 60} pentru Conv1d)

Conv1d(2 → 32, kernel=3, padding=1)  → ReLU → MaxPool1d(2)   [32 × 30]
Conv1d(32 → 64, kernel=3, padding=1) → ReLU → MaxPool1d(2)   [64 × 15]
Conv1d(64 → 128, kernel=3, padding=1)→ ReLU → MaxPool1d(2)   [128 × 7]

Flatten → ℝ^{128 × 7 = 896}

Linear(896 → 2)  →  [τ_raw_fill, τ_raw_exhaust]   (parametri fizici bruti)
```

**Dimensiuni canal:** [32, 64, 128] (configurabil din YAML)

**Ieșire backbone:** 2 valori brute care reprezintă constantele de timp ODE, înainte de activare.

### 2.2 Transformarea la constante de timp fizice

```python
τ_fill    = softplus(τ_raw_fill)    + 0.1      # garantat > 0.1
τ_exhaust = softplus(τ_raw_exhaust) + 0.1
```

`softplus(x) = log(1 + e^x)` — similar cu ReLU dar diferențiabilă în 0, garantează τ > 0.

**De ce offset 0.1?** Previne colapsul fizic τ → 0 care ar face ecuația ODE degenerată.

---

## 3. Funcția de Pierdere Fizică (ODE Loss)

**Fișier:** `utils/physics_loss.py`

### 3.1 Ecuația ODE Pneumatică

Sistemul pneumatic este modelat ca un sistem de ordinul 1:
```
τ · dP/dt + P(t) = P_eq
```
Unde:
- `τ` = constanta de timp [s] — măsoară "viteza" umplerii/evacuării
- `P(t)` = presiunea la momentul t
- `P_eq` = presiunea de echilibru (maximul observat pentru umplere, minimul pentru evacuare)
- `dP/dt ≈ ΔP/Δt` = derivata numerică (diferență finită)

### 3.2 Aplicare selectivă (Masking Fazic)

**Problemă importantă**: Ecuația ODE e validă fizic NUMAI în tranziții, nu în steady-state.
- Când P1 e constant (actuatorul a ajuns la capăt), dP/dt ≈ 0 → orice τ satisface ecuația → degenerare
- Soluție: aplicăm ODE **numai** unde semnalul este monoton

```python
fill_mask    = (dP1 > 0).float()   # P1 crește = faza de umplere activă
exhaust_mask = (dP2 < 0).float()   # P2 scade  = faza de evacuare activă
```

### 3.3 Calculul pierderii

```python
# Reziduuri ODE normalizate la amplitudinea semnalului
res_fill    = (τ_fill    · dP1 + P1_t - P_eq_fill)    / amp1
res_exhaust = (τ_exhaust · dP2 + P2_t - P_eq_exhaust) / amp2

# MSE mascat: media DOAR pe pașii fazei active
loss_fill    = mean(res_fill²    · fill_mask)
loss_exhaust = mean(res_exhaust² · exhaust_mask)

# Regularizare τ: penalizează τ → 0 (colaps fizic)
tau_reg = mean(exp(-τ_fill)) + mean(exp(-τ_exhaust))

# Pierdere totală
L = loss_fill + loss_exhaust + 0.05 · tau_reg
```

**De ce normalizare la amplitudine?** Fără normalizare, pierderea ODE (~2-3) ar domina față de alte pierderi (~0.05) → colaps al gradienților.

### 3.4 Ce înseamnă că τ e discriminativ?

| Clasă | Defect | Efect pe τ |
|---|---|---|
| 0 | Normal | τ_fill ≈ τ_exhaust ≈ valori tipice |
| 1 | Scurgere externă | τ_exhaust ≪ normal (evacuare prea rapidă) |
| 3 | Presiune redusă | τ_fill > normal (umplere mai lentă) |
| 4 | Presiune ridicată | τ_fill < normal (umplere mai rapidă) |
| 5 | Evacuare restricționată | τ_exhaust > normal (evacuare mai lentă) |

---

## 4. Calibrare și Scorul de Anomalie (Mahalanobis)

**Metodă:** `pinn_model.calibrate(class0_loader, device, mode='combined')`

### 4.1 Extragerea celor 9 features

Modelul PINN extrage 9 features din fiecare fereastră, toate derivate din ecuația ODE:

```
Groupe A — Features ODE pure (3):
  1. log(τ_fill)       — constanta de timp la umplere (log pentru stabilitate numerică)
  2. log(τ_exhaust)    — constanta de timp la evacuare
  3. ode_residual      — reziduul ODE mediu (cât de bine urmează semnalul ecuația)

Grup B — Features statistice ale semnalului (6):
  4. amp_P1            — amplitudinea P1 (max-min) în fereastră
  5. amp_P2            — amplitudinea P2
  6. fill_frac         — fracția de pași cu dP1>0 (cât timp durează umplerea)
  7. exhaust_frac      — fracția de pași cu dP2<0
  8. P1_mean           — presiunea medie a camerei 1
  9. P2_mean           — presiunea medie a camerei 2
```

**Notă despre fairness**: Toate cele 9 features sunt derivate din același backbone ODE — features din Grup B sunt calculabile și fără ODE, dar sunt incluse în același vector de features al modelului PINN. Autoencoderdul lucrează direct pe semnalul brut (60×2) și nu are acces la aceste features.

### 4.2 Calibrarea distribuției normale

```python
# 1. Extrage features din TOATE ferestrele normale din train
all_feats = []  # shape: [N_normal, 9]
for batch in class0_loader:
    all_feats.append(model._extract_features(batch))

# 2. Estimează distribuția normală ca Gaussiană multivariată
μ   = mean(all_feats, axis=0)    # centrul distribuției normale [9]
Σ   = cov(all_feats.T) + ε·I    # covarianța + regularizare numerică [9×9]
Σ⁻¹ = inv(Σ)                     # inversa covarianței (stocată pentru inferență)
```

### 4.3 Scorul de anomalie (Distanță Mahalanobis)

```python
δ = features(x) - μ            # deviația față de Normal [batch, 9]
score = δ · Σ⁻¹ · δᵀ          # distanță Mahalanobis la pătrat [batch]
```

**Intuiție geometrică**: Mahalanobis nu este distanța euclidiană standard, ci ține cont de **forma** distribuției normale. Dacă τ_fill variază mult în date normale (std mare), o deviație mare pe τ_fill nu e alarmantă. Dacă P1_mean variază puțin (std mic), orice deviație pe P1_mean e detectată imediat.

**Normal** → features aproape de μ → distanță Mahalanobis mică → scor mic
**Defect** → features depărtate de μ (τ diferit, amplitudini diferite) → distanță mare → scor mare

---

## 5. Autoencoderdul CNN

**Fișier:** `models/autoencoder.py`

### 5.1 Arhitectura Encoder-Decoder

```
Input: x ∈ ℝ^{60 × 2}   (transpus la ℝ^{2 × 60})

─── ENCODER ───
Conv1d(2→32, k=3, p=1)  → ReLU → MaxPool1d(2)   [32 × 30]
Conv1d(32→64, k=3, p=1) → ReLU → MaxPool1d(2)   [64 × 15]
Conv1d(64→128, k=3, p=1)→ ReLU → MaxPool1d(2)   [128 × 7]

─── BOTTLENECK ─── (reprezentare comprimată)
                                                  [128 × 7]  = 896 valori

─── DECODER ───
ConvTranspose1d(128→64, k=2, stride=2)  → ReLU   [64 × 14]
ConvTranspose1d(64→32,  k=2, stride=2)  → ReLU   [32 × 28]
ConvTranspose1d(32→2,   k=2, stride=2)          → [2 × 56]

Interpolate(linear) → [2 × 60]   (60 nu e multiplu de 8 → interpolare la final)

Output: x̂ ∈ ℝ^{60 × 2}   (transpus înapoi)
```

**Note**: 60 / 8 = 7.5 → după 3 pooling de factor 2, avem 7 în loc de 7.5 → dimensiunea nu se recuperează exact → `F.interpolate` reajustează la 60.

### 5.2 Scorul de anomalie al AE

```python
def anomaly_score(self, x):
    x_hat = self.forward(x)          # reconstrucția
    return mean((x - x_hat)², axis=[timp, canale])   # MSE per fereastră
```

**Logică**: Modelul a fost antrenat să reconstruiască semnale normale → pentru un semnal normal, reconstrucția e bună (MSE mic). Pentru un defect (semnal cu τ diferit, amplitudine diferită), reconstrucția e rea (MSE mare) → MSE devine scor de anomalie.

---

## 6. Procedura de Antrenament

### 6.1 Hiperparametri (config_anomaly.yaml)

```yaml
time_steps:         60      # lungimea ferestrei
batch_size:         256     # ferestre per batch
learning_rate:      0.001   # rată de învățare inițială
backbone_channels:  [32, 64, 128]
ae_epochs:          20      # epoci Autoencoder
pinn_anomaly_epochs: 20     # epoci PINN-AD
```

### 6.2 Optimizator și Scheduler

Ambele modele folosesc aceeași strategie de optimizare:

```python
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/20)
```

**Cosine Annealing**: rata de învățare scade de la 0.001 la 0.00005 urmând un cosinus pe parcursul epocilor. Nu scade brusc (cum face StepLR) — permite explorare mai bună la început și convergență fină la final.

```
lr(t) = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·t/T_max))
```

### 6.3 Automatic Mixed Precision (AMP)

Dacă există GPU disponibil:
```python
use_amp    = device.type == 'cuda'
amp_scaler = torch.amp.GradScaler(enabled=use_amp)

# În buclă:
with torch.autocast(device_type='cuda', dtype=torch.float16):
    loss = calculate_loss(model(x), x)
amp_scaler.scale(loss).backward()
amp_scaler.step(optimizer)
```

AMP face calculele în **float16** (jumătate din spațiu, de 2-4× mai rapid pe GPU) dar menține acumulatoarele de gradient în float32 pentru stabilitate numerică.

### 6.4 Gradient Clipping (PINN-AD)

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Previne explodarea gradienților care poate apărea în ODE loss când τ e foarte mic sau reziduu e mare. Norma gradientului total nu poate depăși 1.0.

### 6.5 Bucla de antrenament

**PINN-AD**:
```
Pentru fiecare epocă:
  Pentru fiecare batch x din datele normale:
    1. Forward: backbone(x) → [τ_raw_fill, τ_raw_exhaust]
    2. calculate_physics_loss(params, x) → L_ODE
    3. Backward + gradient clipping + optimizer step
  scheduler.step()
  Print: pierderea medie + LR curent
```

**Autoencoder**:
```
Pentru fiecare epocă:
  Pentru fiecare batch x din datele normale:
    1. Forward: encoder(x) → bottleneck → decoder(x) → x̂
    2. L = MSE(x̂, x)
    3. Backward + optimizer step
  scheduler.step()
  Print: pierderea medie + LR curent
```

---

## 7. Comparație Sintetică

| Aspect | PINN-AD | Autoencoder CNN |
|---|---|---|
| **Parametri totali** | ~120k | ~120k (aceeași arhitectură backbone) |
| **Funcție de pierdere** | Reziduu ODE mascat + regularizare τ | MSE reconstrucție |
| **Scor anomalie** | Distanță Mahalanobis în 9D | MSE de reconstrucție |
| **Prior** | τ · dP/dt + P = P_eq | Niciunul |
| **Interpretabilitate** | Parametri fizici (τ_fill, τ_exhaust) | Eroare reconstrucție (opac) |
| **Date puține** | Robust (AUROC ~0.96 și cu 200 ferestre) | Se degradează rapid sub ~500 ferestre |
| **Zgomot** | Mai robust (features abstracte) | Mai sensibil (comparație pixel cu pixel) |
| **Avantaj principal** | Prior fizic = eficiență de date | Simplitate, nu necesită cunoașterea ODE |

---

## 8. Fluxul complet de rulare

```
1. python scripts/clean_protocol.py
       ↓ generează data/processed/date_protocol_clean.csv

2. python anomaly_main.py
       ↓ antrenează PINN-AD + AE pe date normale
       ↓ calculează AUROC global + per clasă
       ↓ salvează results/anomaly_detection_results.png

3. python anomaly_scarcity_main.py
       ↓ 9 fracții × 5 seed-uri = 45 runde de antrenament
       ↓ salvează results/anomaly_scarcity_results.png

4. python anomaly_noise_main.py
       ↓ 9 niveluri de zgomot × 3 seed-uri = 27 evaluări
       ↓ salvează results/anomaly_noise_results.png
```
