import numpy as np
import pytest
from simulation.runner import MockEnvFields
from env.grid import EnvironmentFields
from env.diffusion import DiffusionGrid

# =====================================================================
# 1. TESTS PARA EL ENTORNO FÍSICO DE DIFUSIÓN DE EDPs REAL (FDM)
# =====================================================================

def test_diffusion_grid_stability():
    """
    Verifica que la simulación de difusión FDM sea numéricamente estable
    (no produce valores NaN ni infinitos) debido al sub-stepping CFL.
    """
    size = (30, 30)
    dx = 1.0
    dt = 0.01  # dt grande para probar el sub-stepping CFL
    
    # Probar con el coeficiente de difusión más alto (Oxígeno, D=2000)
    grid = DiffusionGrid(size=size, dx=dx, dt=dt, D=2000.0, init_val=0.0, boundary_val=1.0)
    
    # Correr 200 pasos de simulación
    for _ in range(200):
        grid.step()
        
    # Verificar que no existan NaNs o Infs
    assert not np.isnan(grid.grid).any(), "La cuadrícula contiene valores NaN"
    assert not np.isinf(grid.grid).any(), "La cuadrícula contiene valores Inf o -Inf"
    assert (grid.grid >= 0.0).all(), "La cuadrícula contiene valores negativos"

def test_mass_conservation_and_consumption():
    """
    Verifica que la masa total de la cuadrícula disminuya inmediatamente tras el consumo
    y que en pasos normales sin agentes la masa no decrezca por causas artificiales.
    """
    env = EnvironmentFields(size=(30, 30), dx=1.0, dt=0.01)
    
    # Inicialmente la rejilla de glucosa tiene 0.0 en todas partes excepto el borde Dirichlet (columna 0)
    initial_mass = np.sum(env.glucose_grid.grid)
    
    # Correr varios pasos de difusión libre (sin consumo)
    for _ in range(10):
        env.step()
        
    post_diffusion_mass = np.sum(env.glucose_grid.grid)
    
    # La masa total debe ser mayor o igual debido al flujo de entrada Dirichlet en el borde izquierdo
    assert post_diffusion_mass >= initial_mass, "La masa disminuyó sin que se aplicara consumo"
    
    # Aplicar consumo localizado
    consume_amount = 0.1
    # Consumir de glucosa en una posición central
    consumed = env.consume(x=15.0, y=15.0, amount=consume_amount, field="glucose")
    assert consumed > 0.0
    
    post_consumption_mass = np.sum(env.glucose_grid.grid)
    
    # La masa total debe haber disminuido exactamente tras el consumo
    assert post_consumption_mass < post_diffusion_mass, "La masa no disminuyó después del consumo"

def test_steady_state_convergence():
    """
    Verifica que el campo de difusión converge a un estado estable sin agentes.
    """
    env = EnvironmentFields(size=(10, 10), dx=1.0, dt=0.01)
    
    converged = False
    tolerance = 1e-4
    max_steps = 1000
    
    for step_idx in range(max_steps):
        prev_grid = env.glucose_grid.grid.copy()
        env.step()
        
        # Calcular el cambio medio absoluto entre pasos sucesivos
        change = np.mean(np.abs(env.glucose_grid.grid - prev_grid))
        
        if change < tolerance:
            converged = True
            break
            
    assert converged, f"El campo no converge a un estado estable dentro de los {max_steps} pasos."

def test_rendering_format():
    """
    Verifica que el método render() devuelva un array RGB de dimensiones válidas
    y formato np.uint8 en el rango [0, 255].
    """
    env = EnvironmentFields(size=(20, 20), dx=1.0, dt=0.01)
    
    # Renderizar glucosa y oxígeno
    img_glucose = env.render(field="glucose")
    img_oxygen = env.render(field="oxygen")
    
    # Verificar forma (ny, nx, 3)
    assert img_glucose.shape == (20, 20, 3)
    assert img_oxygen.shape == (20, 20, 3)
    
    # Verificar tipo uint8
    assert img_glucose.dtype == np.uint8
    assert img_oxygen.dtype == np.uint8
    
    # Verificar que los valores estén en el rango de color de 8 bits
    assert img_glucose.min() >= 0
    assert img_glucose.max() <= 255


# =====================================================================
# 2. TESTS PARA EL ENTORNO FÍSICO DE MOCK SIMPLIFICADO
# =====================================================================

def test_mock_env_fields_initialization():
    """
    Verifica que el entorno físico mock se inicialice con el centro
    del gradiente correcto y una reserva finita y limitada de nutrientes.
    """
    env = MockEnvFields(initial_nutrients=1000.0)
    
    assert np.allclose(env.center, np.array([50.0, 50.0]))
    assert env.nutrients == 1000.0

def test_mock_env_fields_observation():
    """
    Verifica que get_observation en el mock devuelva concentraciones
    coherentes con el gradiente y atenuación de nutrientes.
    """
    env = MockEnvFields(initial_nutrients=2000.0)
    
    obs_center = env.get_observation(50.0, 50.0)
    assert obs_center[0] == 1.0  # Glucosa máxima
    assert obs_center[1] == 0.0  # Sin gradiente en el centro exacto
    
    obs_far = env.get_observation(150.0, 50.0)
    assert obs_far[0] == 0.0  # Glucosa nula por distancia
    assert obs_far[1] > 0.0   # Gradiente activo

def test_mock_env_fields_consumption_depletion():
    """
    Verifica que el consumo reduzca de forma finita la reserva de nutrientes del mock.
    """
    env = MockEnvFields(initial_nutrients=15.0)
    
    consumed = env.consume(50.0, 50.0, amount=10.0)
    assert consumed == 10.0
    assert env.nutrients == 5.0
    
    consumed_over = env.consume(50.0, 50.0, amount=8.0)
    assert consumed_over == 5.0
    assert env.nutrients == 0.0
