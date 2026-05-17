import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PINN_AnomalyDetector(nn.Module):
    """
    PINN antrenat EXCLUSIV pe date normale (clasa 0).
    Nu are cap de clasificare — scopul sau este sa invete parametrii fizici
    tau_fill si tau_exhaust ai dinamicii normale.

    La inferenta: reziduul ODE este scorul de anomalie.
    Semnale cu dinamica diferita de normal → tau nepotrivit → rezidual mare.
    """
    def __init__(self, num_features=2, time_steps=60, backbone_channels=None):
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [32, 64, 128]
        ch1, ch2, ch3 = backbone_channels

        self.conv1 = nn.Conv1d(num_features, ch1, 3, padding=1)
        self.conv2 = nn.Conv1d(ch1, ch2, 3, padding=1)
        self.conv3 = nn.Conv1d(ch2, ch3, 3, padding=1)
        self.relu    = nn.ReLU()
        self.pool    = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()

        self.physics_head = nn.Linear(ch3 * (time_steps // 8), 2)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.pool(self.relu(self.conv1(h)))
        h = self.pool(self.relu(self.conv2(h)))
        h = self.pool(self.relu(self.conv3(h)))
        return self.physics_head(self.flatten(h))  # [batch, 2]: tau_fill, tau_exhaust

    def _ode_residual_per_sample(self, x):
        phys_params  = self.forward(x)
        tau_fill     = F.softplus(phys_params[:, 0]).unsqueeze(1) + 0.1
        tau_exhaust  = F.softplus(phys_params[:, 1]).unsqueeze(1) + 0.1

        P1 = x[:, :, 0]
        P2 = x[:, :, 1]
        dP1 = P1[:, 1:] - P1[:, :-1]
        dP2 = P2[:, 1:] - P2[:, :-1]

        P_eq_fill    = P1.max(1).values.unsqueeze(1)
        P_eq_exhaust = P2.min(1).values.unsqueeze(1)
        amp1 = (P1.max(1).values - P1.min(1).values).unsqueeze(1).detach() + 1e-6
        amp2 = (P2.max(1).values - P2.min(1).values).unsqueeze(1).detach() + 1e-6

        fill_mask    = (dP1 > 0).float()
        exhaust_mask = (dP2 < 0).float()

        res_fill    = (tau_fill    * dP1 + P1[:, 1:] - P_eq_fill)    / amp1
        res_exhaust = (tau_exhaust * dP2 + P2[:, 1:] - P_eq_exhaust) / amp2

        loss_fill    = ((res_fill**2)    * fill_mask).sum(1)    / (fill_mask.sum(1)    + 1e-6)
        loss_exhaust = ((res_exhaust**2) * exhaust_mask).sum(1) / (exhaust_mask.sum(1) + 1e-6)

        return loss_fill + loss_exhaust  # [batch]

    def calibrate(self, class0_loader, device):
        """
        Calculeaza centrul distributiei tau pe date normale (clasa 0).
        Scorul de anomalie devine distanta Mahalanobis fata de acest centru.
        """
        taus = []
        self.eval()
        with torch.no_grad():
            for x_scaled, _, _ in class0_loader:
                x_scaled = x_scaled.to(device)
                tf, te = self.get_tau(x_scaled)
                taus.append(torch.stack([tf, te], dim=1).cpu())
        taus = torch.cat(taus, dim=0).numpy()  # [N, 2]

        self.tau_mean = taus.mean(axis=0)       # [2]
        cov = np.cov(taus.T) + np.eye(2) * 1e-6
        self.tau_cov_inv = np.linalg.inv(cov)   # [2, 2]

    def anomaly_score(self, x):
        """Distanta Mahalanobis in spatiul (tau_fill, tau_exhaust) fata de centrul normal."""
        with torch.no_grad():
            tf, te = self.get_tau(x)
            taus = torch.stack([tf, te], dim=1).cpu().numpy()  # [batch, 2]
            delta = taus - self.tau_mean                        # [batch, 2]
            # score[i] = delta[i] @ cov_inv @ delta[i].T
            scores = np.einsum('bi,ij,bj->b', delta, self.tau_cov_inv, delta)
            return torch.tensor(scores, dtype=torch.float32)

    def get_tau(self, x):
        with torch.no_grad():
            phys_params = self.forward(x)
            return (F.softplus(phys_params[:, 0]) + 0.1,
                    F.softplus(phys_params[:, 1]) + 0.1)
