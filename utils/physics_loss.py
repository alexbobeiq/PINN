import torch
import torch.nn.functional as F

def calculate_physics_loss(phys_params, x_scaled):
    """
    ODE pneumatic de ordinul 1 cu masking fazic:

        tau * dP/dt + P = P_eq

    Aplicat SELECTIV pe fazele monotone:
      - tau_fill    pe dP1 > 0  (camera 1 se umple)
      - tau_exhaust pe dP2 < 0  (camera 2 se evacuează)

    Masking-ul elimina degenerescenta din implementarile anterioare:
    ecuatia e valida fizic DOAR in tranzitii, nu in steady-state (dP≈0).
    Astfel tau_fill si tau_exhaust devin discriminative intre clase:
      Clasa 1: tau_exhaust >> normal  (ex_0=300, evacuare blocata)
      Clasa 3: tau_fill > normal      (pre_input=1650, umplere lenta)
      Clasa 4: tau_fill < normal      (pre_input=4000, umplere rapida)
      Clasa 5: tau_exhaust intermediar (ex_0=400)
    """
    # Tau-uri pozitive cu offset minim fizic rezonabil
    tau_fill    = F.softplus(phys_params[:, 0]).unsqueeze(1) + 0.1  # [batch, 1]
    tau_exhaust = F.softplus(phys_params[:, 1]).unsqueeze(1) + 0.1  # [batch, 1]

    P1 = x_scaled[:, :, 0]   # [batch, T]
    P2 = x_scaled[:, :, 1]   # [batch, T]

    dP1 = P1[:, 1:] - P1[:, :-1]   # [batch, T-1]
    dP2 = P2[:, 1:] - P2[:, :-1]

    P1_t = P1[:, 1:]   # P la pasul curent (aliniat cu derivata)
    P2_t = P2[:, 1:]

    # P_eq ancorate la extremele observabile ale ferestrei:
    #   P1 tinde spre maximul sau (umplere)
    #   P2 tinde spre minimul sau (evacuare)
    # Ancorare din semnal → fara scurgere de label
    P_eq_fill    = P1.max(dim=1).values.unsqueeze(1)   # [batch, 1]
    P_eq_exhaust = P2.min(dim=1).values.unsqueeze(1)   # [batch, 1]

    # Amplitudine per-sample: normalizeaza reziduurile ca sa fie adimensionale
    # Fara normalizare, physics_loss ~ 2-3 >> cross_entropy ~ 0.05 → colaps
    amp1 = (P1.max(dim=1).values - P1.min(dim=1).values).unsqueeze(1).detach() + 1e-6
    amp2 = (P2.max(dim=1).values - P2.min(dim=1).values).unsqueeze(1).detach() + 1e-6

    # Masking fazic: ODE aplicat DOAR unde semnalul e monoton
    fill_mask    = (dP1 > 0).float()   # P1 creste = faza umplere activa
    exhaust_mask = (dP2 < 0).float()   # P2 scade  = faza evacuare activa

    # Reziduuri ODE normalizate: tau * dP/dt + P - P_eq = 0, impartit la amplitudine
    res_fill    = (tau_fill    * dP1 + P1_t - P_eq_fill)    / amp1
    res_exhaust = (tau_exhaust * dP2 + P2_t - P_eq_exhaust) / amp2

    # MSE mascat: media DOAR pe pasii fazei active
    n_fill    = fill_mask.sum(dim=1)    + 1e-6
    n_exhaust = exhaust_mask.sum(dim=1) + 1e-6

    loss_fill    = ((res_fill    ** 2) * fill_mask).sum(dim=1)    / n_fill
    loss_exhaust = ((res_exhaust ** 2) * exhaust_mask).sum(dim=1) / n_exhaust

    loss_ode = (loss_fill + loss_exhaust).mean()

    # Regularizare tau: penalizeaza tau → 0 (colaps fizic)
    tau_reg = torch.mean(torch.exp(-tau_fill)) + torch.mean(torch.exp(-tau_exhaust))

    return loss_ode + 0.05 * tau_reg
