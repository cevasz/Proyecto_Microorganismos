import pytest
import numpy as np
import torch
from simulation.colony import Colony
from simulation.runner import MockEnvFields
from modules.neural_ode.pi_node import PINode
from modules.lstm_division.model import DivisionLSTM

def test_colony_initialization():
    """
    Verifica que la colonia se inicialice con el número de agentes solicitado,
    instanciando y compartiendo los modelos neuronales una única vez.
    """
    env = MockEnvFields()
    colony = Colony(initial_agents=15, env_fields=env)
    
    # Cantidad inicial de agentes
    assert len(colony.agents) == 15
    
    # Modelos centralizados compartidos en Colony
    assert isinstance(colony.neural_ode, PINode)
    assert isinstance(colony.division_model, DivisionLSTM)
    
    # Modelos en modo de evaluación
    assert not colony.neural_ode.training
    assert not colony.division_model.training

def test_colony_step():
    """
    Verifica que Colony.step ejecute en bloque los modelos neuronales,
    actualice todos los estados y maneje adecuadamente el ciclo de vida.
    """
    env = MockEnvFields()
    colony = Colony(initial_agents=10, env_fields=env)
    
    # Registrar estados bioquímicos iniciales
    init_states = torch.stack([a.internal_state.clone() for a in colony.agents])
    
    # Avanzar un paso temporal
    colony.step(dt=0.01)
    
    # Los estados bioquímicos de todos los agentes deben haber cambiado tras la ODE
    new_states = torch.stack([a.internal_state for a in colony.agents])
    assert not torch.equal(init_states, new_states)
    
    # La cantidad de agentes no debe desbordarse inmediatamente
    assert len(colony.agents) == 10

def test_colony_lifecycle_and_division():
    """
    Verifica que la colonia elimine linealmente a los agentes muertos
    y multiplique a los agentes divididos.
    """
    env = MockEnvFields()
    colony = Colony(initial_agents=3, env_fields=env)
    
    # Caso 1: Forzar muerte de una bacteria
    colony.agents[0].energy = 0.0
    colony.agents[0].is_dead = True
    
    # Caso 2: Forzar división de otra bacteria
    colony.agents[1].length = 2.0
    colony.agents[1].is_divided = True
    
    # Avanzar paso
    colony.step(dt=0.01)
    
    # Población final esperada: 3 agentes - 1 muerto - 1 dividido + 2 hijos = 3 agentes
    assert len(colony.agents) == 3
    # Ninguna bacteria viva debe estar marcada como muerta o dividida
    for agent in colony.agents:
        assert agent.is_dead is False
        assert agent.is_divided is False

def test_colony_stats():
    """
    Verifica que get_population_stats calcule correctamente la densidad,
    centro de gravedad espacial y recuento de agentes.
    """
    env = MockEnvFields()
    colony = Colony(initial_agents=0, env_fields=env)
    
    # Estadísticas para colonia vacía
    empty_stats = colony.get_population_stats()
    assert empty_stats["N"] == 0
    assert empty_stats["avg_density"] == 0.0
    assert empty_stats["spatial_center"] == (0.0, 0.0)
    
    # Añadir agentes controlados
    from agents.bacterium import Bacterium
    a1 = Bacterium(position=(10.0, 10.0))
    a2 = Bacterium(position=(20.0, 20.0))
    colony.agents.extend([a1, a2])
    
    stats = colony.get_population_stats()
    assert stats["N"] == 2
    assert np.isclose(stats["spatial_center"][0], 15.0)
    assert np.isclose(stats["spatial_center"][1], 15.0)
    assert stats["avg_density"] > 0.0
