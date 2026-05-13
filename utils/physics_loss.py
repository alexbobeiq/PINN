import torch

def calculate_physics_loss(phys_pred, inputs):
    # Index 2: Setpoint Alimentare (2500)
    # Index 3: Grad Evacuare (10000)
    sp_alim = inputs[:, -1, 2] 
    ex_0 = inputs[:, -1, 3]
    
    # NORMALIZARE: Împărțim la 2500 pentru ca reziduul să fie mic (ordinul 0-1)
    # Target-ul fizic devine 1.0 (pentru regim normal)
    target_fizic = (sp_alim / 2500.0) * (ex_0 / 10000.0)
    predictie_scalata = phys_pred.squeeze() / 2500.0
    
    reziduu = predictie_scalata - target_fizic
    
    return torch.mean(reziduu**2)