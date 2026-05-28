import torch
import numpy as np
from collections import deque

class Bacterium:
    """
    Agente individual que representa una bacteria E. coli.
    Entidad ligera que almacena el estado fisiológico y cinemático.
    La inferencia de modelos neuronales se realiza de forma vectorizada a nivel de Colony.
    """
    def __init__(self, position=(0.0, 0.0)):
        # --- Atributos Fisiológicos y Cinemáticos ---
        self.position = list(position)
        self.orientation = np.random.uniform(0, 2 * np.pi)
        self.age = 0
        self.length = 1.0
        self.energy = 1.0
        self.metabolic_rate = 0.05  # Tasa de consumo base por segundo
        
        # --- Vector de estado z para Neural ODE ---
        # [CheY-P, ppGpp, metilación_receptores, energía_ATP]
        self.internal_state = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        
        # Flags de ciclo de vida
        self.is_divided = False
        self.is_dead = False
        self.history = deque(maxlen=60)  # Buffer eficiente O(1) para entrada de LSTM (máx 60 muestras)
        self.pos_history = deque(maxlen=10) # Buffer para estelas de movimiento (trails)
        self.last_action = 0

        # Constantes de movimiento biológico
        self.run_speed = 25.0  # μm/s

    def update_kinematics_and_metabolism(self, action, env_fields, dt=0.01):
        """
        Actualiza el estado cinemático (Run/Tumble), consumo de energía,
        metabolismo y buffer de historial para LSTM basándose en la acción decidida.
        """
        self.age += int(dt * 1000)
        self.last_action = action
        self.pos_history.append(list(self.position))
        # Cinemática: 0 = Run, 1 = Tumble
        if action == 0:
            # Ejecuta Run (avance recto)
            self.position[0] = float(np.clip(self.position[0] + np.cos(self.orientation) * self.run_speed * dt, 0.0, 100.0))
            self.position[1] = float(np.clip(self.position[1] + np.sin(self.orientation) * self.run_speed * dt, 0.0, 100.0))
        else:
            # Ejecuta Tumble (cambio abrupto de orientación aleatorio)
            self.orientation += np.random.normal(0, np.pi/2) 
            
        # Consumo metabólico e impacto ambiental
        nutriente_consumido = env_fields.consume(self.position[0], self.position[1], self.metabolic_rate * dt)
        if nutriente_consumido > 0:
            # Multiplicador aumentado a 4.0 para corregir el balance de energía: comer permite recuperar energía y tener ganancia neta
            self.energy = min(1.0, self.energy + (nutriente_consumido * 4.0) - (self.metabolic_rate * dt))
        else:
            self.energy -= (self.metabolic_rate * dt)
            
        self.internal_state[3] = self.energy  # Sincronizar energía al vector matemático ODE
        if self.energy <= 0.0:
            self.is_dead = True
            return
            
        # Crecimiento celular heurístico
        self.length += (0.01 * dt)
            
        # Registrar en el buffer de historial para LSTM
        # Al usar collections.deque(maxlen=60), descarta los elementos viejos en O(1) de manera automática
        dna_rate = min(self.age / 1200000.0, 1.0)
        self.history.append([self.length, self.energy, dna_rate])

    def divide(self):
        """
        Invocado por el simulador global (Colony). 
        Genera dos agentes descendientes y reparte biomasa y estado interno.
        """
        # Clonación de posición con jitter espacial gaussiano
        pos1 = [self.position[0] + np.random.normal(0, 0.5), self.position[1] + np.random.normal(0, 0.5)]
        pos2 = [self.position[0] + np.random.normal(0, 0.5), self.position[1] + np.random.normal(0, 0.5)]
                
        hijo1 = Bacterium(position=pos1)
        hijo2 = Bacterium(position=pos2)
        
        # Herencia Biológica (División mitótica simulada + error replicativo)
        ruido1 = torch.randn(4) * 0.05
        ruido2 = torch.randn(4) * 0.05
        hijo1.internal_state = torch.clamp((self.internal_state / 2.0) + ruido1, 0.0, 1.0)
        hijo2.internal_state = torch.clamp((self.internal_state / 2.0) + ruido2, 0.0, 1.0)
        
        # Reparto de volumen y energía base
        hijo1.length = self.length / 2.0
        hijo2.length = self.length / 2.0
        hijo1.energy = self.energy / 2.0
        hijo2.energy = self.energy / 2.0
        
        return [hijo1, hijo2]
