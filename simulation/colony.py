import numpy as np
import torch
from agents.bacterium import Bacterium
from modules.neural_ode.pi_node import PINode
from modules.lstm_division.model import DivisionLSTM

class Colony:
    """
    Gestor principal del sistema multiobjetivo (Multi-agent system).
    Administra la lista de agentes Bacterium, su ciclo de vida y centraliza
    la ejecución de modelos neuronales (PINode, DivisionLSTM, PPO) de manera vectorizada.
    """
    def __init__(self, initial_agents=10, env_fields=None, rl_policy=None):
        self.agents = []
        self.env_fields = env_fields
        self.rl_policy = rl_policy
        
        # Centralización de modelos neuronales en la colonia (instanciados una sola vez)
        self.neural_ode = PINode(state_dim=4, hidden_dim=64)
        self.division_model = DivisionLSTM(input_dim=3, hidden_dim=128, num_layers=2, dropout=0.2)
        
        # Poner los modelos en modo de evaluación para desactivar dropout/etc y acelerar
        self.neural_ode.eval()
        self.division_model.eval()
        
        # Spawn inicial
        for _ in range(initial_agents):
            pos = (np.random.uniform(0, 100), np.random.uniform(0, 100))
            self.agents.append(Bacterium(position=pos))

    def step(self, dt=0.01):
        """
        Avanza la simulación ejecutando los modelos de IA de forma vectorizada en bloque,
        y actualizando la cinemática, metabolismo y reproducción de la colonia de forma eficiente.
        """
        if len(self.agents) == 0:
            return
            
        # 1. NEURAL ODE: Integración de la dinámica biológica interna de todos los agentes en batch
        z0 = torch.stack([agent.internal_state for agent in self.agents])  # [N, 4]
        t_span = torch.tensor([0.0, dt], dtype=torch.float32)
        
        with torch.no_grad():
            z_trajectory = self.neural_ode(z0, t_span)  # [2, N, 4]
            z_next = z_trajectory[-1]  # [N, 4]
            
        for i, agent in enumerate(self.agents):
            agent.internal_state = z_next[i]
            
        # 2. OBSERVAR Y DECIDIR ACCIONES (RL POLICY)
        obs_env_list = []
        for agent in self.agents:
            obs_env = self.env_fields.get_observation(agent.position[0], agent.position[1])
            obs_env_list.append(obs_env)
            
        obs_env_tensor = torch.tensor(obs_env_list, dtype=torch.float32)  # [N, 4]
        internal_obs = z_next[:, :2]  # [N, 2]  (CheY-P y ppGpp)
        full_obs_tensor = torch.cat([obs_env_tensor, internal_obs], dim=-1)  # [N, 6]
        
        if self.rl_policy is not None:
            # Detectar el tamaño de observación del modelo cargado
            obs_shape = self.rl_policy.observation_space.shape[0]
            if obs_shape == 4:
                # El modelo preentrenado PPO (Fase 1) espera [pos_x, pos_y, grad_x, grad_y]
                obs_rl_list = []
                for agent in self.agents:
                    # Gradiente ficticio hacia el centro (50, 50) como en MockEnvFields
                    grad = np.array([50.0 - agent.position[0], 50.0 - agent.position[1]])
                    norm = np.linalg.norm(grad)
                    if norm > 0:
                        grad = grad / norm
                    obs_rl_list.append([agent.position[0], agent.position[1], grad[0], grad[1]])
                full_obs_np = np.array(obs_rl_list, dtype=np.float32)
                actions_raw, _ = self.rl_policy.predict(full_obs_np, deterministic=True)
                
                # Mapear las acciones continuas a discretas (0 = Run, 1 = Tumble)
                # En BacteriumEnv, la segunda dimensión del vector de acción es tumble_probability
                actions = []
                for act in actions_raw:
                    tumble_prob = np.clip((act[1] + 1.0) / 2.0, 0.0, 1.0)
                    if np.random.random() < tumble_prob:
                        actions.append(1)  # Tumble
                    else:
                        actions.append(0)  # Run
                actions = np.array(actions)
            else:
                # El modelo espera [obs_env (4) + internal_obs (2)] = 6
                full_obs_np = full_obs_tensor.numpy()
                actions, _ = self.rl_policy.predict(full_obs_np, deterministic=True)
        else:
            actions = np.random.choice([0, 1], size=len(self.agents))
            
        # 3. KINEMATICS & METABOLISM: Actualizar física de cada bacteria
        for i, agent in enumerate(self.agents):
            agent.update_kinematics_and_metabolism(actions[i], self.env_fields, dt)
            
        # 4. DIVISION CELULAR (LSTM) en bloque para agentes elegibles
        eligible_agents = [agent for agent in self.agents if not agent.is_dead and len(agent.history) == 60]
        if len(eligible_agents) > 0:
            x_seq = torch.tensor([agent.history for agent in eligible_agents], dtype=torch.float32)
            with torch.no_grad():
                p_div_batch = self.division_model(x_seq)  # [M, 1]
                
            for idx, agent in enumerate(eligible_agents):
                if p_div_batch[idx].item() > 0.85:
                    agent.is_divided = True
                    
        # 5. GESTIÓN DEL CICLO DE VIDA (Nacimientos y Muertes)
        new_agents = []
        dead_agents = []
        
        for agent in self.agents:
            if agent.is_dead:
                dead_agents.append(agent)
            elif agent.is_divided:
                hijos = agent.divide()
                new_agents.extend(hijos)
                dead_agents.append(agent)
                
        # Remoción y adición segura
        for da in dead_agents:
            if da in self.agents:
                self.agents.remove(da)
                
        self.agents.extend(new_agents)
        
    def get_population_stats(self):
        """
        Retorna estadísticas globales macroscópicas de la colonia poblacional.
        """
        N = len(self.agents)
        if N == 0:
            return {"N": 0, "avg_density": 0.0, "spatial_center": (0.0, 0.0)}
            
        positions = np.array([agent.position for agent in self.agents])
        
        # Densidad media (agentes por área ocupada del bounding box)
        if N > 1:
            dx = np.max(positions[:, 0]) - np.min(positions[:, 0])
            dy = np.max(positions[:, 1]) - np.min(positions[:, 1])
            area = dx * dy
            avg_density = N / (area + 1e-5)
        else:
            avg_density = 0.0
            
        spatial_center = (np.mean(positions[:, 0]), np.mean(positions[:, 1]))
        
        return {
            "N": N,
            "avg_density": avg_density,
            "spatial_center": spatial_center
        }
