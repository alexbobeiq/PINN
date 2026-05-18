# Documentație Experimente — Detecție Anomalii PINN vs Autoencoder

## Contextul general

Toate cele 3 experimente urmăresc aceeași întrebare centrală:

> *Poate un model fizic (PINN) să detecteze defecte în sistemul pneumatic VTEM mai bine decât un model pur de date (Autoencoder CNN), și în ce condiții este avantajul cel mai mare?*

**Principiu de bază:** Ambele modele sunt antrenate **exclusiv pe date Normale** (clasa 0, sistem funcțional).
La inferență, un exemplu cu scor de anomalie mare este clasificat ca defect.
Nu se folosesc etichete de defect la antrenament — aceasta este detecție de anomalii, nu clasificare.

---

## Pregătirea datelor (comună tuturor experimentelor)

### Fișierul sursă
`data/processed/date_protocol_clean.csv` — generat de `scripts/clean_protocol.py` din datele brute de protocol PLC.

### Ce conține
| Coloană | Semnificație |
|---|---|
| `Presiune1_Valva1` | Presiunea camerei 1 (senzor, în unități PLC) |
| `Presiune2_Valva1` | Presiunea camerei 2 |
| `pre_input` | Setpoint presiune alimentare (variabil per clasă) |
| `ex_0` | Setpoint evacuare (variabil per clasă) |
| `Label` | 0=Normal, 1–7=tipuri de defect |
| `segment` | ID segment continuu (fără gap-uri temporale) |

### Clase de defect
| Label | Defect | Setpoint modificat |
|---|---|---|
| 0 | Normal | pre_input=2463, ex_0=10000 |
| 1 | Scurgere externă Valva1 | ex_0=300 (evacuare rapidă) |
| 2 | Scurgere internă V1↔V2 | ex_0=300 (idem) |
| 3 | Presiune redusă (1650) | pre_input=1650 (~67% nominal) |
| 4 | Presiune ridicată (4000) | pre_input=4000 (~163% nominal) |
| 5 | Evacuare restricționată (400) | ex_0=400 |
| 6 | Întârziere retragere (1s) | Temporizare PLC modificată |
| 7 | Eroare senzor (+500) | +500 adăugat la Presiune1 în post-procesare |

### Preprocesare
1. **Underflow uint16→int16**: valori ≥32768 sunt convertite la negative (ex: 65534→-2)
2. **Gap-uri temporale**: discontinuitățile >1s din timestamp sunt detectate; 5 rânduri înainte și după fiecare gap sunt eliminate pentru a evita ferestre corupte
3. **Clasa 7**: offsetul de +500 nu era aplicat în modul protocol PLC → adăugat manual în CSV la post-procesare
4. **Normalizare**: `StandardScaler` pe cele 2 coloane de presiune (medie 0, deviație standard 1)

### Fereastra temporală (sliding window)
- Dimensiune: **T = 60 de pași** (~1.2 secunde la ~20ms/pas)
- O fereastră `[i, i+T)` este validă doar dacă toate cele T eșantioane aparțin aceluiași segment (fără gap intern)
- Eticheta ferestrei = eticheta ultimului eșantion din fereastră

### Împărțire train/test (leave-segment-out)
- Protocolul PLC a rulat de mai multe ori → datele sunt organizate în **segmente** continue
- Pentru fiecare clasă: ultimele 20% din segmente merg la **test**, restul la **train**
- Nu există ferestre care traversează granițele train/test → nu există data leakage

---

## Experiment 1: Detecție Anomalii — Comparație de Bază

**Script:** `anomaly_main.py` → `evaluate/anomaly_detection.py::run_anomaly_experiment()`

### Obiectiv
Comparăm PINN-AD și Autoencoder CNN pe **toată mulțimea de date disponibile**.
Metrica principală: **AUROC** (Area Under ROC Curve) pentru Normal vs Orice Defect.
AUROC = 1.0 înseamnă separare perfectă; AUROC = 0.5 înseamnă aleator.

### Pașii experimentului

#### Pasul 1: Filtram datele normale din train
```
Din train_loader, selectăm DOAR ferestrele cu Label=0 (clasa Normal).
Acestea formează class0_loader — singurul date pe care modelele vor fi antrenate.
```

#### Pasul 2: Antrenăm PINN-AD
```
train_pinn_anomaly(class0_loader, config, device)
    → ODE loss pe date normale, 20 epoci
    → Modelul învață să parametrizeze τ_fill și τ_exhaust pentru un ciclu normal
```

#### Pasul 3: Antrenăm Autoencoderdul CNN
```
train_autoencoder(class0_loader, config, device)
    → MSE reconstruction loss pe date normale, 20 epoci
    → Modelul învață să comprime și să reconstruiască un ciclu normal
```

#### Pasul 4: Calibrare PINN (Mahalanobis)
```
pinn_model.calibrate(class0_loader, device, mode='combined')
    → Extrage 9 features din toate ferestrele normale
    → Calculează media μ și inversa covarianței Σ⁻¹ în spațiul de 9D
```

#### Pasul 5: Colectăm scoruri pe setul de test
- **PINN**: scor = distanță Mahalanobis față de distribuția normală calibrată
- **AE**: scor = MSE de reconstrucție (eroare mai mare = mai anormal)

#### Pasul 6: Calculăm AUROC
- Global: Normal vs toate defectele
- Per clasă: Normal vs fiecare defect separat

### Rezultat tipic
| Model | AUROC global |
|---|---|
| PINN-AD (9 features ODE) | **~0.96** |
| Autoencoder CNN | ~0.89 |

---

## Experiment 2: Scarcity — Robustețe la Date Puține

**Script:** `anomaly_scarcity_main.py` → `evaluate/anomaly_detection.py::run_anomaly_scarcity_experiment()`

### Motivație
Scenariul real de implementare industrială: operatorul are acces la **puține date de funcționare normală** înainte să instaleze sistemul de monitorizare.

**Ipoteza de demonstrat:**
> *PINN-ul menține AUROC ridicat și cu puține date normale, deoarece fizica ODE structurează spațiul de parametri chiar și cu 200 de ferestre. Autoencoderdul, fără un astfel de prior, se degradează rapid.*

### Structura experimentului

```
fractions = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
n_seeds   = 5   (repetăm cu 5 inițializări aleatoare diferite pentru stabilitate statistică)
```

Pentru fiecare fracție:
1. Din cele ~N_normal ferestre normale din train, selectăm aleator `max(32, int(N_normal × frac))` ferestre
2. Antrenăm un PINN-AD și un AE **de la zero** pe acest subset redus
3. Evaluăm AUROC pe setul de test **complet** (neschimbat)
4. Repetăm de 5 ori cu seed-uri diferite și raportăm medie ± deviație standard

### De ce nu scade PINN-ul?
Fizica ODE constrânge modelul: chiar cu 200 de ferestre normale, rețeaua CNN backbone extrage τ-uri consistente, iar distribuția normală în spațiul (τ_fill, τ_exhaust, ode_res, ...) este bine definită. Covarianța pe 9 features cu 200 de sample-uri este stabilă (regula practică: ai nevoie de ~10× dimensiunea featurelor, adică ~90 de sample-uri).

AE, fără prior fizic, supraînvață pe puținele date → reconstrucția devine haotică → pragul de anomalie nu mai separă Normal de Defect.

### Vizualizare
Grafic cu 2 subploturi:
- **Stânga**: AUROC vs Număr ferestre (scală logaritmică), cu banda de ±1σ
- **Dreapta**: ΔAUROC = PINN − AE (avantajul absolut al PINN)

---

## Experiment 3: Robustețe la Zgomot

**Script:** `anomaly_noise_main.py` → `evaluate/anomaly_detection.py::run_anomaly_noise_experiment()`

### Motivație
Senzorii industriali introduc zgomot de măsurare. Vrem să știm care model este mai robust:
- PINN-AD: extrage τ și reziduuri ODE — features mai "abstracte", mai puțin sensibile la zgomot punctual
- AE: compară pixeli de semnal — direct afectat de zgomot pe fiecare eșantion

### Structura experimentului

```
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]  (σ în unități normalizate)
n_seeds      = 3
```

**Important**: Antrenamentul se face pe date **curate** (fără zgomot). Zgomotul este adăugat **doar la inferență**.
Aceasta simulează scenariul real: modelul e calibrat pe date bune, dar la deployment senzorii pot fi zgomotoși.

Zgomot adăugat la inferență:
```python
x_noisy = x_scaled + torch.randn_like(x_scaled) * sigma
```
`sigma=0.0` = fără zgomot (baseline), `sigma=2.0` = zgomot masiv (semnalul aproape ininteligibil).

### Pașii experimentului
1. Antrenăm n_seeds perechi (PINN-AD, AE) pe date normale **curate**
2. Calibrăm fiecare PINN pe date curate
3. Pentru fiecare nivel de zgomot σ:
   - Adăugăm zgomot Gaussian la fiecare batch de test
   - Calculăm AUROC pentru PINN și AE
   - Raportăm medie ± std peste seed-uri

### Vizualizare
Grafic cu 3 subploturi:
- **Stânga**: AUROC vs σ, cu benzi de incertitudine
- **Centru**: ΔAUROC = PINN − AE (verde dacă PINN câștigă, roșu dacă AE câștigă)
- **Dreapta**: Degradare relativă față de σ=0 în procente (mai mic = mai robust)

---

## Metrica: AUROC

**De ce AUROC și nu Accuracy?**

AUROC nu depinde de alegerea unui prag. Modelele de anomalie returnează un **scor continuu** — nu știm a priori ce prag să folosim. AUROC măsoară cât de bine **ordonează** modelul exemplele: toate exemplele defecte ar trebui să aibă scor mai mare decât toate exemplele normale, indiferent de prag.

- AUROC = 1.0 → separare perfectă
- AUROC = 0.5 → model aleator (nu știe să separe)
- AUROC = 0.9 → dacă iei un exemplu defect aleator și unul normal aleator, modelul îl va scora corect pe cel defect ca mai anormal în 90% din cazuri

**Formula AUROC** (integrala de sub curba ROC):
```
AUROC = P(score(defect) > score(normal))
```
