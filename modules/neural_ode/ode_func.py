import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    Red MLP que implementa dz/dt = f(z, t; θ) para la simulación de E. coli.
    Vector de estado z = [CheY-P, ppGpp, metilación_receptores, energía_ATP]
    (Valores normalizados a [0, 1])
    """
    def __init__(self, state_dim=4, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        
        # Arquitectura MLP con activación Tanh (diferenciable para solver ODE)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, t, z):
        """
        t: Escalar de tiempo.
        z: Vector de estado (CheY-P, ppGpp, metilación, energía)
        Retorna dz/dt de dimensión state_dim.
        """
        return self.net(z)
