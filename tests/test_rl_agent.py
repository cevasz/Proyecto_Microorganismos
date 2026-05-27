import pytest
import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

from env.environment import BacteriumEnv
from modules.rl_agent.baselines import RandomWalker
from modules.rl_agent.agent import ChemotaxisLoggingCallback, train_ppo_agent

def test_random_walker_baseline():
    """
    Verifica que el agente RandomWalker se inicialice correctamente y genere acciones
    válidas dentro de los límites y formas del espacio de acciones.
    """
    env = BacteriumEnv()
    walker = RandomWalker(env.action_space)
    
    obs, info = env.reset()
    
    # Generar y validar 50 acciones
    for _ in range(50):
        action = walker.act(obs)
        assert action.shape == (2,)
        assert action.dtype == np.float32
        
        # Verificar límites físicos
        assert env.action_space.contains(action), f"Acción fuera de límites: {action}"
        
        # Ejecutar acción
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

def test_ppo_integration_short():
    """
    Realiza una corrida de integración extremadamente corta (80 pasos) con PPO
    para validar que las importaciones, dimensiones de observación, VecEnv paralelos
    y callbacks funcionen end-to-end sin errores de tiempo de ejecución.
    """
    config_path = "configs/default.yaml"
    
    # Verificar que el archivo de configuración por defecto existe
    assert os.path.exists(config_path), "No se encontró el configs/default.yaml"
    
    # Ejecutar entrenamiento ultracorto (16 pasos en paralelo en 8 envs = 128 timesteps globales)
    # Esto asegura que la recopilación de datos y paso de Gymnasium -> SB3 es correcto.
    train_ppo_agent(config_path=config_path, total_timesteps=128)
    
    # Verificar que se crearon los directorios de checkpoints y el modelo final
    assert os.path.exists("models/checkpoints/"), "No se creó el directorio de checkpoints"
    assert os.path.exists("models/ppo_chemotaxis_final.zip"), "No se guardó el modelo PPO final"
    
    # Limpiar modelo final del test de integración para no ensuciar el espacio de trabajo
    try:
        os.remove("models/ppo_chemotaxis_final.zip")
    except OSError:
        pass
