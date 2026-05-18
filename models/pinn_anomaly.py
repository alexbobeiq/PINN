import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PINN_AnomalyDetector(nn.Module):
    """
    PINN antrenat EXCLUSIV pe date normale (clasa 0).
    Scorul de anomalie = distanta Mahalanobis in spatiul de 9 caracteristici fizice:
      [log_tau_fill, log_tau_exhaust,
       amp_P1, amp_P2,
       fill_frac, exhaust_frac,
       ode_residual_adaptiv,
       P1_mean, P2_mean]

    log_tau      – constante de timp (log pentru a stabiliza τ extreme)
    amp          – amplitudinile de presiune (prinde schimbarile de setpoint)
    fill/exhaust – fractia de pasi cu faza activa (prinde valve blocate)
    ode_residual – reziduul ODE cu τ adaptiv (prinde scurgeri: termen extra decay)
    P_mean       – presiunea medie (prinde deviatii de setpoint clasa 3/4)
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
        return self.physics_head(self.flatten(h))   # [batch, 2]: raw params

    def get_tau(self, x):
        with torch.no_grad():
            phys_params = self.forward(x)
            return (F.softplus(phys_params[:, 0]) + 0.1,
                    F.softplus(phys_params[:, 1]) + 0.1)

    def _physics_features(self, x):
        """
        3 features pur fizice (PINN-specifice):
          [log_tau_fill, log_tau_exhaust, ode_residual_adaptiv]
        Acestea nu pot fi calculate fara modelul de fizica.
        """
        tf, te = self.get_tau(x)

        P1 = x[:, :, 0]; P2 = x[:, :, 1]
        dP1 = P1[:, 1:] - P1[:, :-1]
        dP2 = P2[:, 1:] - P2[:, :-1]

        amp1_n = (P1.max(1).values - P1.min(1).values).unsqueeze(1) + 1e-6
        amp2_n = (P2.max(1).values - P2.min(1).values).unsqueeze(1) + 1e-6
        P_eq_fill    = P1.max(1).values.unsqueeze(1)
        P_eq_exhaust = P2.min(1).values.unsqueeze(1)
        fill_mask    = (dP1 > 0).float()
        exhaust_mask = (dP2 < 0).float()
        res_fill    = (tf.unsqueeze(1) * dP1 + P1[:, 1:] - P_eq_fill)    / amp1_n
        res_exhaust = (te.unsqueeze(1) * dP2 + P2[:, 1:] - P_eq_exhaust) / amp2_n
        lf = ((res_fill    ** 2) * fill_mask).sum(1)    / (fill_mask.sum(1)    + 1e-6)
        le = ((res_exhaust ** 2) * exhaust_mask).sum(1) / (exhaust_mask.sum(1) + 1e-6)

        feats = torch.stack([torch.log(tf), torch.log(te), lf + le], dim=1)
        return feats.cpu().numpy()

    def _signal_features(self, x):
        """
        6 features statistice de semnal (non-fizice, orice metoda le poate calcula):
          [amp_P1, amp_P2, fill_frac, exhaust_frac, P1_mean, P2_mean]
        Folosite separat ca baseline pentru comparatie corecta in paper.
        """
        P1 = x[:, :, 0]; P2 = x[:, :, 1]
        dP1 = P1[:, 1:] - P1[:, :-1]
        dP2 = P2[:, 1:] - P2[:, :-1]
        feats = torch.stack([
            P1.max(1).values - P1.min(1).values,
            P2.max(1).values - P2.min(1).values,
            (dP1 > 0).float().mean(1),
            (dP2 < 0).float().mean(1),
            P1.mean(1),
            P2.mean(1),
        ], dim=1)
        return feats.cpu().numpy()

    def _extract_features(self, x):
        """
        Combina features fizice + statistice de semnal (9 total).
        NOTA: pentru comparatie corecta in paper, foloseste calibrate(mode='physics').
        """
        return np.concatenate([self._physics_features(x),
                               self._signal_features(x)], axis=1)

    def calibrate(self, class0_loader, device, mode='physics'):
        """
        Invata distributia Normala si stocheaza Mahalanobis.

        mode='physics'  → 3 features pur fizice [log_τf, log_τe, ode_res]
                          Comparatie corecta cu AE (pentru paper).
        mode='combined' → 9 features (fizice + statistici semnal)
                          Performanta maxima, dar nu e fair vs AE singur.
        """
        self.calibration_mode = mode
        self.eval()
        all_feats = []
        with torch.no_grad():
            for x_scaled, _, _ in class0_loader:
                x_scaled = x_scaled.to(device)
                f = (self._physics_features(x_scaled) if mode == 'physics'
                     else self._extract_features(x_scaled))
                all_feats.append(f)

        all_feats = np.concatenate(all_feats, axis=0)
        self.feat_mean    = all_feats.mean(axis=0)
        cov               = np.cov(all_feats.T) + np.eye(all_feats.shape[1]) * 1e-6
        self.feat_cov_inv = np.linalg.inv(cov)
        self.tau_mean     = np.exp(self.feat_mean[:2])

        names = (['log_τf', 'log_τe', 'ode_res'] if mode == 'physics'
                 else ['log_τf', 'log_τe', 'ode_res', 'amp1', 'amp2',
                       'fill_fr', 'exh_fr', 'P1_μ', 'P2_μ'])
        vals = ' | '.join(f'{n}={v:.3f}' for n, v in zip(names, self.feat_mean))
        print(f"  [Calibrare Normal, mode={mode}] {vals}")

    def anomaly_score(self, x):
        """
        Distanta Mahalanobis in spatiul features calibrat (physics sau combined).
        Normal → distanta mica;  Defect → distanta mare.
        """
        with torch.no_grad():
            mode = getattr(self, 'calibration_mode', 'physics')
            feats = (self._physics_features(x) if mode == 'physics'
                     else self._extract_features(x))
            delta = feats - self.feat_mean
            score = np.einsum('bi,ij,bj->b', delta, self.feat_cov_inv, delta)
            return torch.tensor(score, dtype=torch.float32)
