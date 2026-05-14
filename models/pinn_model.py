import torch
import torch.nn as nn

class PINN_Classifier(nn.Module):
    def __init__(self, num_features=2, num_classes=8, time_steps=60,
                 backbone_channels=None):
        super(PINN_Classifier, self).__init__()

        if backbone_channels is None:
            backbone_channels = [16, 32, 64]
        ch1, ch2, ch3 = backbone_channels

        self.conv1 = nn.Conv1d(num_features, ch1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(ch1,          ch2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(ch2,          ch3, kernel_size=3, padding=1)
        self.relu    = nn.ReLU()
        self.pool    = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        conv_out_dim = ch3 * (time_steps // 8)

        # Physics head: invata [tau_fill, tau_exhaust] constransi exclusiv de ODE loss
        self.physics_head = nn.Linear(conv_out_dim, 2)

        # Clasificator: features CNN + 2 tau fizici (fara analytical features)
        self.fc1 = nn.Linear(conv_out_dim + 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.pool(self.relu(self.conv1(h)))
        h = self.pool(self.relu(self.conv2(h)))
        h = self.pool(self.relu(self.conv3(h)))
        features = self.flatten(h)

        phys_params = self.physics_head(features)   # [batch, 2]: tau_fill, tau_exhaust

        combined = torch.cat([features, phys_params], dim=1)
        logits = self.fc2(self.relu(self.fc1(combined)))

        return logits, phys_params