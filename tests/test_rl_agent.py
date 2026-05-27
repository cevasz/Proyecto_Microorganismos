import pytest
import os
import numpy as np
import gymnasium as gym
from env.environment import BacteriumEnv
from modules.rl_agent.baselines import RandomWalker
from modules.rl_agent.agent import ChemotaxisLoggingCallback, train_ppo_agent

def test_random_walker_baseline():
    """
    Verifica que el agente RandomWalker se inicialice correctamente y genere acciones
    válidas dentro de los límites y dimensiones del espacio de acciones reales [-1.0, 1.0].
    """
    env = BacteriumEnv()
    walker = RandomWalker(env.action_space)
    
    obs, info = env.reset()
    
    # Generar y validar 20 acciones aleatorias
    for _ in range(20):
        action = walker.act(obs)
        assert action.shape == (2,)
        assert action.dtype == np.float32
        
        # Verificar que la acción se encuentre estrictamente dentro de los límites Gym
        assert env.action_space.contains(action), f"Acción fuera de límites: {action}"
        
        # Ejecutar en el entorno
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

def test_ppo_integration_short():
    """
    Realiza una corrida extremadamente corta de entrenamiento PPO para verificar
    que las integraciones, callbacks de logging, VecEnvs paralelos y guardados
    de checkpoints funcionen end-to-end sin problemas de ejecución.
    """
    config_path = "configs/default.yaml"
    
    # Asegurar la existencia del archivo de configuración
    assert os.path.exists(config_path), "No se encontró el configs/default.yaml"
    
    # Copiar el modelo final preexistente si existe para no borrarlo en el test
    backup_exists = os.path.exists("models/ppo_chemotaxis_final.zip")
    if backup_exists:
        os.rename("models/ppo_chemotaxis_final.zip", "models/ppo_chemotaxis_final.zip.bak")
        
    try:
        # Ejecutar entrenamiento ultracorto (16 pasos por cada uno de los 8 entornos paralelos = 128 timesteps globales)
        # Esto valida que Gymnasium y Stable-Baselines3 interactúen con total estabilidad.
        train_ppo_agent(config_path=config_path, total_timesteps=128)
        
        assert os.path.exists("models/checkpoints/"), "No se creó el directorio de checkpoints"
        assert os.path.exists("models/ppo_chemotaxis_final.zip"), "No se guardó el modelo PPO final"
        
    finally:
        # Limpieza del modelo de prueba y restauración del backup del modelo original
        if os.path.exists("models/ppo_chemotaxis_final.zip"):
            try:
                os.remove("models/ppo_chemotaxis_final.zip")
            except OSError:
                pass
                
        if backup_exists:
            if os.path.exists("models/ppo_chemotaxis_final.zip.bak"):
                os.rename("models/ppo_chemotaxis_final.zip.bak", "models/ppo_chemotaxis_final.zip")
