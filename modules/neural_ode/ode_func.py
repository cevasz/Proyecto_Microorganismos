import torch
import torch.nn as nn

class CheYODEFunc(nn.Module):
    """
    Módulo 2: Neural ODE (Prototipo Fase 1 MVP)
    Aprende la dinámica f(z, t; \theta) para la señalización CheA/CheY.
    """
    def __init__(self, state_dim=4, hidden_dim=64):
        super().__init__()
        # Red neuronal que define la derivada temporal dz/dt
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, t, z):
        # z: [CheY-P, ppGpp, metilación, energía]
        return self.net(z)
