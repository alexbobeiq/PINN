import torch
import torch.nn.functional as F

def calculate_physics_loss(phys_pred, x_scaled):
    """
    Physical State Reconstruction Loss.
    Fortam reteaua sa reconstruiasca integral din spatiul latent toata curba de presiuni.
    Deoarece datele sunt scalate standard, acest MSE are natural o scara de ~1.0, 
    deci nu e ignorat de optimizator (cum se intampla cu derivatele care aveau scara 0.01).
    """
    # Aplatizam starea reala pentru a o compara cu predictia
    true_physics = x_scaled.reshape(x_scaled.size(0), -1) # [batch, 120]
    
    # MSE loss obliga spatiul latent sa pastreze informatia intregii curbe fizice
    return F.mse_loss(phys_pred, true_physics)