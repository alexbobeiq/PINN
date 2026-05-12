import torch

def calculate_physics_loss(phys_pred, inputs):
    # Extragem valorile din tensor la ultimul pas de timp:
    # 0: P1, 1: P2, 2: Setpoint Alimentare, 3: Grad Evacuare (ex_0)
    p1 = inputs[:, -1, 0]
    p2 = inputs[:, -1, 1]
    ex_0 = inputs[:, -1, 3]
    
    # Un exemplu de ecuație de constrângere a debitului/presiunii:
    # Dacă evacuarea (ex_0) e sugrumată, diferența de presiune P1-P2 dictează forța.
    # Penalizăm rețeaua dacă 'phys_pred' deviază de la comportamentul teoretic.
    reziduu = phys_pred.squeeze() - ((p1 - p2) * ex_0)
    
    return torch.mean(reziduu**2)
