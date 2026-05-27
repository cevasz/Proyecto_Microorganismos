import torch
import numpy as np
import pytest
from agents.bacterium import Bacterium
from simulation.runner import MockEnvFields

def test_bacterium_initialization():
    """
    Verifica que el agente Bacterium se inicialice con los valores fisiológicos
    y cinemáticos correctos, incluyendo la deque de historial O(1).
    """
    agent = Bacterium(position=(10.0, 20.0))
    
    assert agent.position == [10.0, 20.0]
    assert 0 <= agent.orientation <= 2 * np.pi
    assert agent.age == 0
    assert agent.length == 1.0
    assert agent.energy == 1.0
    assert agent.metabolic_rate == 0.05
    
    # Estado bioquímico interno para Neural ODE [CheY-P, ppGpp, metilación, energía]
    assert agent.internal_state.shape == (4,)
    assert agent.internal_state.dtype == torch.float32
    assert agent.internal_state[3] == 1.0  # Energía inicializada a 1.0
    
    # Buffer de historial eficiente
    assert len(agent.history) == 0
    assert agent.history.maxlen == 60
    assert agent.is_dead is False
    assert agent.is_divided is False

def test_bacterium_kinematics_and_metabolism():
    """
    Verifica que update_kinematics_and_metabolism aplique correctamente los desplazamientos
    según la acción elegida, descuente energía y actualice el buffer de historial.
    """
    agent = Bacterium(position=(50.0, 50.0))
    env = MockEnvFields()
    
    # Fijar orientación hacia la derecha para calcular movimiento exacto (0)
    agent.orientation = 0.0
    
    # Paso 1: Ejecutar Run (acción = 0)
    # velocidad = 25 um/s, dt = 0.01s -> desplazamiento = 0.25 um hacia X
    agent.update_kinematics_and_metabolism(action=0, env_fields=env, dt=0.01)
    
    assert np.isclose(agent.position[0], 50.25)
    assert np.isclose(agent.position[1], 50.0)
    assert agent.age == 10  # 0.01s * 1000 = 10ms
    assert len(agent.history) == 1
    
    # Paso 2: Ejecutar Tumble (acción = 1)
    # Debe alterar la orientación estocásticamente, pero no cambiar la posición
    old_pos = list(agent.position)
    old_orientation = agent.orientation
    
    agent.update_kinematics_and_metabolism(action=1, env_fields=env, dt=0.01)
    
    assert agent.position == old_pos
    assert agent.orientation != old_orientation
    assert len(agent.history) == 2

def test_bacterium_starvation_and_death():
    """
    Verifica que la bacteria muera de inanición si su energía decae por debajo o igual a 0.
    """
    agent = Bacterium(position=(0.0, 0.0))  # Completamente fuera del radio de comida
    env = MockEnvFields()
    agent.energy = 0.0001
    
    # Forzar metabolismo y verificar muerte
    agent.update_kinematics_and_metabolism(action=0, env_fields=env, dt=0.1)
    
    assert agent.is_dead is True

def test_bacterium_division():
    """
    Verifica que la bipartición genere dos bacterias hijas con el reparto
    equitativo de biomasa y energía, y ruido estocástico en su estado interno.
    """
    agent = Bacterium(position=(30.0, 30.0))
    agent.length = 2.0
    agent.energy = 0.8
    agent.internal_state = torch.tensor([0.4, 0.6, 0.2, 0.8], dtype=torch.float32)
    
    hijos = agent.divide()
    
    assert len(hijos) == 2
    h1, h2 = hijos
    
    # Posición con jitter
    assert np.linalg.norm(np.array(h1.position) - np.array(agent.position)) < 5.0
    assert np.linalg.norm(np.array(h2.position) - np.array(agent.position)) < 5.0
    
    # Reparto de biomasa y energía
    assert h1.length == 1.0
    assert h2.length == 1.0
    assert h1.energy == 0.4
    assert h2.energy == 0.4
    
    # Estado interno heredado con ruido
    assert h1.internal_state.shape == (4,)
    assert h2.internal_state.shape == (4,)
    assert not torch.equal(h1.internal_state, h2.internal_state)  # Ruido estocástico
