import numpy as np
import pytest
from simulation.runner import MockEnvFields

def test_mock_env_fields_initialization():
    """
    Verifica que el entorno físico simulado se inicialice con el centro
    del gradiente correcto y una reserva finita y limitada de nutrientes.
    """
    env = MockEnvFields(initial_nutrients=1000.0)
    
    assert np.allclose(env.center, np.array([50.0, 50.0]))
    assert env.nutrients == 1000.0

def test_mock_env_fields_observation():
    """
    Verifica que get_observation devuelva concentraciones de glucosa altas
    cerca del centro, decayendo al alejarse, con gradientes apuntando al centro.
    """
    env = MockEnvFields(initial_nutrients=2000.0)
    
    # Observación en el centro (máxima concentración)
    obs_center = env.get_observation(50.0, 50.0)
    # [glucosa, gradiente_g, oxigeno, gradiente_o]
    assert obs_center[0] == 1.0  # Glucosa máxima
    assert obs_center[1] == 0.0  # Sin gradiente en el centro exacto
    assert obs_center[2] == 1.0  # Oxígeno
    
    # Observación alejada
    obs_far = env.get_observation(150.0, 50.0)
    assert obs_far[0] == 0.0  # Glucosa nula por distancia
    assert obs_far[1] > 0.0   # Gradiente activo apuntando a la fuente

def test_mock_env_fields_consumption_depletion():
    """
    Verifica que el consumo por parte de las bacterias reduzca de forma finita
    la reserva de nutrientes del entorno, y que no se puedan consumir más de lo disponible.
    """
    env = MockEnvFields(initial_nutrients=15.0)
    
    # Consumir menos del total disponible en la zona rica
    consumed = env.consume(50.0, 50.0, amount=10.0)
    assert consumed == 10.0
    assert env.nutrients == 5.0
    
    # Consumir más del restante disponible
    consumed_over = env.consume(50.0, 50.0, amount=8.0)
    assert consumed_over == 5.0  # Solo otorga los 5.0 restantes
    assert env.nutrients == 0.0
    
    # Consumir sin recursos restantes
    consumed_none = env.consume(50.0, 50.0, amount=5.0)
    assert consumed_none == 0.0

def test_mock_env_fields_attenuation_decay():
    """
    Verifica que el gradiente químico percibido en get_observation decaiga
    y se atenúe gradualmente conforme los nutrientes se agotan, cayendo a 0
    al estar totalmente agotada la reserva.
    """
    env = MockEnvFields(initial_nutrients=500.0)
    
    # Sensación a media reserva
    obs_half = env.get_observation(50.0, 50.0)
    assert obs_half[0] == 0.5  # Atenuado al 50%
    
    # Vaciar reserva
    env.consume(50.0, 50.0, amount=500.0)
    
    # Sensación sin recursos
    obs_empty = env.get_observation(50.0, 50.0)
    assert obs_empty[0] == 0.0  # Glucosa completamente nula
    assert obs_empty[1] == 0.0  # Gradiente nulo
