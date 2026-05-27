import numpy as np
import pytest
import gymnasium as gym
from env.environment import BacteriumEnv

def test_environment_spaces():
    """
    Verifica que los espacios de observación y acción de BacteriumEnv estén
    definidos de acuerdo a las especificaciones reales de la Fase 1.
    """
    env = BacteriumEnv()
    
    # 1. Espacio de Observación: [pos_x, pos_y, grad_x, grad_y]
    assert env.observation_space.shape == (4,)
    assert env.observation_space.dtype == np.float32
    
    # 2. Espacio de Acción: [run_duration, tumble_probability]
    assert env.action_space.shape == (2,)
    assert env.action_space.dtype == np.float32
    
    # Verificar límites de acción del wrapper Gym [-1.0, 1.0]
    assert np.allclose(env.action_space.low, np.array([-1.0, -1.0], dtype=np.float32))
    assert np.allclose(env.action_space.high, np.array([1.0, 1.0], dtype=np.float32))

def test_environment_reset():
    """
    Verifica el correcto funcionamiento de reset() y la inicialización de variables.
    """
    env = BacteriumEnv()
    obs, info = env.reset()
    
    # Verificar formato de retorno de Gymnasium
    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)
    
    # Verificar inicialización de paso
    assert env.step_count == 0
    assert env.accumulated_reward == 0.0
    assert len(env.run_durations) == 0
    assert len(env.is_running) == 0

def test_environment_step():
    """
    Verifica que el método step() avance el estado del entorno, aplique las acciones
    escaladas y devuelva el formato estándar de Gymnasium.
    """
    env = BacteriumEnv()
    obs, info = env.reset()
    
    # Acción: Acción válida dentro de [-1.0, 1.0]
    action = np.array([0.0, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    # Validaciones de tipos y formas
    assert next_obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Verificar incremento de pasos
    assert env.step_count == 1
    assert "accumulated_reward" in info
    assert "mean_run_duration" in info
    assert "run_fraction" in info
    assert "net_distance" in info
