import numpy as np
import pytest
import gymnasium as gym
from env.environment import BacteriumEnv

def test_environment_spaces():
    """
    Verifica que los espacios de observación y acción estén definidos correctamente
    según las especificaciones del espacio de estado y acción.
    """
    env = BacteriumEnv()
    
    # 1. Espacio de Observación
    assert env.observation_space.shape == (8,)
    assert env.observation_space.dtype == np.float32
    
    # 2. Espacio de Acción
    assert env.action_space.shape == (2,)
    assert env.action_space.dtype == np.float32
    # Verificar límites de acción
    assert np.allclose(env.action_space.low, np.array([0.1, 0.0], dtype=np.float32))
    assert np.allclose(env.action_space.high, np.array([3.0, 1.0], dtype=np.float32))

def test_environment_reset():
    """
    Verifica el correcto funcionamiento del método reset() y la inicialización
    de variables de estado, historia y la observación inicial.
    """
    env = BacteriumEnv()
    obs, info = env.reset()
    
    # Verificar formato de retorno estándar Gymnasium
    assert obs.shape == (8,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)
    
    # Verificar que el agente esté centrado inicialmente
    assert np.allclose(env.bacterium.position, [100.0, 100.0])
    
    # Verificar que el historial de glucosa se inicializa correctamente con valores no nulos
    # (dado que la columna 0 es Dirichlet=1.0, en el centro [100, 100] inicialmente es 0.0)
    assert len(env.glucose_history) == 3
    assert env.glucose_history == [0.0, 0.0, 0.0]
    
    # El estado de energía de la bacteria debe inicializarse en 1.0 (máximo)
    assert np.isclose(env.bacterium.internal_state[3].item(), 1.0)

def test_environment_step_run():
    """
    Verifica el comportamiento de una acción 'Run' pura (p_tumble = 0.0).
    Debe desplazar la bacteria física en su dirección actual theta por tau_run segundos,
    consumir nutrientes, y descontar energía por correr.
    """
    env = BacteriumEnv()
    env.reset()
    
    # Fijar theta para predecir el movimiento exacto (hacia la derecha, theta=0)
    env.theta = 0.0
    
    # Acción: Correr por 1.0 segundo con probabilidad 0 de tumble
    action = np.array([1.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Verificar salidas estándar
    assert obs.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # La posición debe haberse desplazado exactamente run_speed * tau_run hacia la derecha (eje X)
    # velocidad = 25.0, tiempo = 1.0 -> desplazamiento = 25.0
    expected_x = 100.0 + 25.0
    expected_y = 100.0
    assert np.isclose(env.bacterium.position[0], expected_x)
    assert np.isclose(env.bacterium.position[1], expected_y)
    
    # El costo energético de correr es cost_run * tau_run (0.04 * 1.0 = 0.04)
    # La bacteria partió con 1.0 de energía, y en esa zona no hay glucosa (concentración es 0.0),
    # por lo tanto, no hay ganancia de energía. Su energía final debe ser aproximadamente 1.0 - 0.04 = 0.96.
    assert np.isclose(env.bacterium.internal_state[3].item(), 0.96)
    
    # El reward debe ser alpha * glucosa_consumida - gamma * costo_energetico
    # glucosa_consumida = 0.0, costo_energetico = 0.04, gamma = 0.1, alpha = 1.0
    # reward = -0.1 * 0.04 = -0.004
    assert np.isclose(reward, -0.1 * 0.04)

def test_environment_step_tumble():
    """
    Verifica el comportamiento de una acción 'Tumble' pura (p_tumble = 1.0).
    Debe reorientar estocásticamente a la bacteria, no desplazar su posición física,
    y aplicar un costo energético de tumble.
    """
    env = BacteriumEnv()
    env.reset()
    
    old_theta = env.theta
    old_pos = list(env.bacterium.position)
    
    # Acción: Tumble puro (p_tumble = 1.0)
    action = np.array([1.5, 1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # La posición física no debería variar
    assert np.allclose(env.bacterium.position, old_pos)
    
    # El ángulo theta debe haberse reorientado de forma estocástica (casi seguro es diferente)
    # Nota: hay una probabilidad infima de que sea igual, pero conceptualmente es una nueva dirección.
    assert env.theta != old_theta
    
    # La duración del paso debió ser tumble_duration = 0.1 s
    assert info["dt_elapsed"] == 0.1
    assert info["tumbled"] is True
    
    # El costo de energía para tumble es cost_tumble * tumble_duration = 0.08 * 0.1 = 0.008
    # Energía final: 1.0 - 0.008 = 0.992
    assert np.isclose(env.bacterium.internal_state[3].item(), 0.992)
