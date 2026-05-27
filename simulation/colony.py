import numpy as np

class Colony:
    """
    Gestor principal del sistema multiobjetivo (Multi-agent system).
    Administra la lista de agentes Bacterium, su ciclo de vida y la recolección de métricas.
    """
    def __init__(self, initial_agents=10, env_fields=None, rl_policy=None):
        self.agents = []
        self.env_fields = env_fields
        self.rl_policy = rl_policy
        
        # Import local para evitar posibles bucles de dependencia
        from agents.bacterium import Bacterium
        
        # Spawn inicial
        for _ in range(initial_agents):
            pos = (np.random.uniform(0, 100), np.random.uniform(0, 100))
            self.agents.append(Bacterium(position=pos, rl_policy=self.rl_policy))

    def step(self, dt=0.01):
        """
        Avanza la simulación iterando sobre toda la colonia.
        Gestiona explícitamente los nacimientos (fisones) y muertes (inanición).
        """
        new_agents = []
        dead_agents = []
        
        for agent in self.agents:
            agent.step(self.env_fields, dt)
            
            if agent.is_dead:
                dead_agents.append(agent)
            elif agent.is_divided:
                hijos = agent.divide()
                new_agents.extend(hijos)
                dead_agents.append(agent) # El organismo parental desaparece biológicamente
                
        # Remoción segura de registros muertos/divididos
        for da in dead_agents:
            if da in self.agents:
                self.agents.remove(da)
                
        # Incorporación de nuevos agentes
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
