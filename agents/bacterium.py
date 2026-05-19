import torch
from modules.neural_ode.pi_node import PINode
from modules.lstm_division.model import DivisionLSTM

class Bacterium:
    """
    Agente individual que representa una bacteria E. coli.
    Punto de integración para los 3 Módulos:
    - Módulo 1 (Dev A): RL Agent (a inyectar desde el Environment/Runner)
    - Módulo 2 (Dev B): Neural ODE (CheA/CheY)
    - Módulo 3 (Dev B): LSTM (División Celular)
    """
    def __init__(self, position=(0.0, 0.0)):
        self.position = list(position)
        self.steps_lived = 0
        
        # Módulo 2: Estado interno simulado por Neural ODE
        self.internal_state = torch.tensor([0.0, 0.0, 0.0, 1.0]) # [CheY-P, ppGpp, metilación, energía]
        self.neural_ode = PINode(state_dim=4, hidden_dim=64)
        
        # Módulo 3: Predicción de división por LSTM
        self.division_model = DivisionLSTM()
        self.is_divided = False
        
    def update(self, dt=0.01):
        """
        Actualiza el estado interno de la bacteria por un paso de tiempo dt.
        """
        self.steps_lived += 1
        
        # Simular avance de 1 paso en la Neural ODE (MVP)
        t = torch.tensor([0.0, dt])
        # En la Fase 2 aquí usaríamos torchdiffeq.odeint(self.neural_ode, self.internal_state, t)
        dz = self.neural_ode(t[1], self.internal_state)
        self.internal_state = self.internal_state + dz * dt
        
        # Verificar si la célula debe dividirse
        if self.division_model.predict_division_stub(self.steps_lived):
            self.is_divided = True
            
    def reset_division(self):
        """
        Reinicia el contador y el estado de división para la célula hija.
        """
        self.steps_lived = 0
        self.is_divided = False
        # Mitad de energía tras división
        self.internal_state[3] /= 2.0 

