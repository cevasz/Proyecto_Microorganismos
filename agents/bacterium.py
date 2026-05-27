import torch
import numpy as np
import copy
from modules.neural_ode.pi_node import PINode
from modules.lstm_division.model import DivisionLSTM

class Bacterium:
    """
    Agente individual que representa una bacteria E. coli.
    Punto de integración maestro para los Módulos RL, Neural ODE y LSTM.
    """
    def __init__(self, position=(0.0, 0.0), rl_policy=None):
        # --- Atributos Fisiológicos y Cinemáticos ---
        self.position = list(position)
        self.orientation = np.random.uniform(0, 2 * np.pi)
        self.age = 0
        self.length = 1.0
        self.energy = 1.0
        self.metabolic_rate = 0.05  # Tasa de consumo base por segundo
        
        # --- Módulo 2: Neural ODE (PI-NODE) ---
        # Vector de estado z: [CheY-P, ppGpp, metilación_receptores, energía_ATP]
        self.internal_state = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self.neural_ode = PINode(state_dim=4, hidden_dim=64)
        
        # --- Módulo 3: LSTM (División Celular) ---
        self.division_model = DivisionLSTM(input_dim=3, hidden_dim=128, num_layers=2, dropout=0.2)
        
        # --- Módulo 1: RL Agent (Policy) ---
        self.rl_policy = rl_policy
        
        # Flags de ciclo de vida
        self.is_divided = False
        self.is_dead = False
        self.history = []  # Buffer LSTM
        
        # Constantes de movimiento biológico
        self.run_speed = 25.0  # μm/s
        
    def step(self, env_fields, dt=0.01):
        """
        Ciclo de vida principal invocado en cada iteración por el simulador de la colonia.
        """
        self.age += int(dt * 1000)
        
        # 1. OBSERVAR: Entorno 2D (Responsabilidad Dev A)
        # Se asume que get_observation retorna [glucosa, grad_g, o2, grad_o2]
        obs_env = env_fields.get_observation(self.position[0], self.position[1])
        obs_env_tensor = torch.tensor(obs_env, dtype=torch.float32)
        
        # 2. NEURAL ODE: Integración de la dinámica biológica interna
        t_span = torch.tensor([0.0, dt], dtype=torch.float32)
        z_trajectory = self.neural_ode(self.internal_state, t_span)
        self.internal_state = z_trajectory[-1]
        
        # =====================================================================
        # DECISIÓN DE DISEÑO: ACOPLAMIENTO RL <-> NEURAL ODE
        # =====================================================================
        # En biología real, E. coli no percibe gradientes ambientales de forma "mágica" u omnisciente.
        # Los ligandos externos se acoplan a receptores de transmembrana, gatillando 
        # una cascada de fosforilación que altera los niveles de CheA y CheY-P.
        # Por lo tanto, el nivel interno de CheY-P (y ppGpp) actúa como una memoria química
        # de corto plazo; es un "filtro integrador" que encapsula la historia de concentraciones
        # a las que la célula ha sido expuesta.
        # 
        # Al concatenar el output predictivo de la Neural ODE (z_next[:2] correspondiente 
        # a CheY-P y ppGpp) con la observación cruda del agente RL, forzamos a que la red de
        # política (PPO) base sus decisiones de Run/Tumble en el propio ESTADO BIOQUÍMICO
        # simulado de la célula. Esto elimina la limitación de los agentes reactivos simples
        # y cierra magistralmente el bucle de retroalimentación físico-computacional.
        # =====================================================================
        internal_obs = self.internal_state[:2]  # [CheY-P, ppGpp]
        full_obs = torch.cat([obs_env_tensor, internal_obs], dim=-1)
        
        # 3. RL POLICY: Decisión y Cinemática
        if self.rl_policy is not None:
            # Predicción con modelo SB3 (espera numpy array)
            action, _ = self.rl_policy.predict(full_obs.numpy(), deterministic=True)
        else:
            action = np.random.choice([0, 1]) # 0 = Run, 1 = Tumble
            
        if action == 0:
            # Ejecuta Run (avance recto)
            self.position[0] += np.cos(self.orientation) * self.run_speed * dt
            self.position[1] += np.sin(self.orientation) * self.run_speed * dt
        else:
            # Ejecuta Tumble (cambio abrupto de orientación aleatorio)
            self.orientation += np.random.normal(0, np.pi/2) 
            
        # 4. CONSUMO metabólico e impacto ambiental
        # Asume que .consume() retorna la cantidad de nutriente efectivamente captada
        nutriente_consumido = env_fields.consume(self.position[0], self.position[1], self.metabolic_rate * dt)
        if nutriente_consumido > 0:
            self.energy = min(1.0, self.energy + (nutriente_consumido * 0.1))
        else:
            self.energy -= (self.metabolic_rate * dt)
            
        self.internal_state[3] = self.energy # Sincronizar energía al vector matemático ODE
        if self.energy <= 0.0:
            self.is_dead = True
            return
            
        # Crecimiento heurístico continuo
        self.length += (0.01 * dt)
            
        # 5. LSTM: Inferencia de División Celular
        dna_rate = min(self.age / 1200000.0, 1.0)
        self.history.append([self.length, self.energy, dna_rate])
        
        if len(self.history) > 60:
            self.history.pop(0)
            
        # 6. ACTUALIZAR flag de división
        if len(self.history) == 60:
            x_seq = torch.tensor([self.history], dtype=torch.float32)
            with torch.no_grad():
                p_div = self.division_model(x_seq).item()
            
            if p_div > 0.85:
                self.is_divided = True

    def divide(self):
        """
        Invocado por el simulador global. Genera dos agentes descendientes y reparte biomasa.
        """
        # Clonación de posición con jitter espacial gaussiano
        pos1 = [self.position[0] + np.random.normal(0, 0.5), self.position[1] + np.random.normal(0, 0.5)]
        pos2 = [self.position[0] + np.random.normal(0, 0.5), self.position[1] + np.random.normal(0, 0.5)]
                
        hijo1 = Bacterium(position=pos1, rl_policy=self.rl_policy)
        hijo2 = Bacterium(position=pos2, rl_policy=self.rl_policy)
        
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
