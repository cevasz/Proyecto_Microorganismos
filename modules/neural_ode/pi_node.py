import torch
import torch.nn as nn
from torchdiffeq import odeint
from .ode_func import ODEFunc

class PINode(nn.Module):
    """
    Physics-Informed Neural ODE (PI-NODE).
    Envoltorio que integra ODEFunc y calcula las penalizaciones físicas.
    """
    def __init__(self, state_dim=4, hidden_dim=64, vmax=1.0, km=0.5, lambda_mass=0.1, lambda_nn=0.1, lambda_mm=0.1):
        super().__init__()
        self.ode_func = ODEFunc(state_dim, hidden_dim)
        
        # Parámetros biológicos (Michaelis-Menten)
        self.vmax = vmax
        self.km = km
        
        # Pesos de penalización
        self.lambda_mass = lambda_mass
        self.lambda_nn = lambda_nn
        self.lambda_mm = lambda_mm

    def forward(self, z0, t_span):
        """
        Integración de la trayectoria con torchdiffeq.
        z0: tensor de estado inicial.
        t_span: tensor con el rango de tiempo (ej. [0.0, dt]).
        Retorna z_trajectory.
        """
        z_trajectory = odeint(self.ode_func, z0, t_span, method='dopri5')
        return z_trajectory

    def compute_loss(self, z_pred, z_true):
        """
        Pérdida total con MSE (observación) y Física (Physics-Informed).
        """
        # 1. Error de Reconstrucción (MSE Loss)
        mse_loss = nn.MSELoss()(z_pred, z_true)
        
        # Computar derivadas aprendidas para evaluar la física (no dependen de t en E.coli base)
        dz_dt = self.ode_func(None, z_pred)
        
        # 2. Conservación de masa: suma(dz/dt) ≈ 0 -> L2
        mass_loss = torch.mean(torch.sum(dz_dt, dim=-1)**2)
        
        # 3. No negatividad: penalizar valores donde z < 0 -> ReLU(-z)
        negativity_loss = torch.mean(torch.relu(-z_pred)**2)
        
        # 4. Cinética de Michaelis-Menten (CheY-P): dCheYP/dt <= Vmax * CheYP / (Km + CheYP)
        chey_p = z_pred[..., 0]
        dchey_p_dt = dz_dt[..., 0]
        mm_bound = self.vmax * chey_p / (self.km + chey_p + 1e-8)
        
        # Penaliza si dCheYP/dt supera la cota física (dchey_p_dt - mm_bound > 0)
        mm_loss = torch.mean(torch.relu(dchey_p_dt - mm_bound)**2)
        
        # Sumatoria final
        physics_loss = (self.lambda_mass * mass_loss + 
                        self.lambda_nn * negativity_loss + 
                        self.lambda_mm * mm_loss)
                        
        total_loss = mse_loss + physics_loss
        
        return total_loss, mse_loss, physics_loss
