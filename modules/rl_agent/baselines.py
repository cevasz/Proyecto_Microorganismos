import numpy as np

class RandomWalker:
    """
    Agente de comparación (baseline) que toma decisiones completamente aleatorias
    dentro del espacio de acciones del entorno BacteriumEnv.
    Permite contrastar estadísticamente (ej. Mann-Whitney U test) el desempeño de PPO.
    """
    def __init__(self, action_space):
        """
        Args:
            action_space (gymnasium.spaces.Box): Espacio de acciones del entorno.
        """
        self.action_space = action_space

    def act(self, obs=None):
        """
        Genera una acción aleatoria de manera uniforme dentro de los límites del espacio de acciones.
        
        Args:
            obs (np.ndarray): Observación actual del entorno (no se utiliza).
            
        Returns:
            np.ndarray: Acción aleatoria [tau_run, p_tumble].
        """
        return self.action_space.sample()
