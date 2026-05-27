import numpy as np
import pytest
from env.grid import EnvironmentFields
from env.diffusion import DiffusionGrid

def test_diffusion_grid_stability():
    """
    Verifica que la simulación de difusión FDM sea numéricamente estable
    (no produce valores NaN ni infinitos) debido al sub-stepping CFL.
    """
    # Usar dimensiones típicas del default.yaml
    size = (50, 50)
    dx = 1.0
    dt = 0.01  # dt grande para probar el sub-stepping CFL
    
    # Probar con el coeficiente de difusión más alto (Oxígeno, D=2000)
    grid = DiffusionGrid(size=size, dx=dx, dt=dt, D=2000.0, init_val=0.0, boundary_val=1.0)
    
    # Correr 1000 pasos de simulación
    for _ in range(1000):
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
    # Por lo tanto, la difusión desde el borde Dirichlet inyectará masa al sistema.
    initial_mass = np.sum(env.glucose_grid.grid)
    
    # Correr varios pasos de difusión libre (sin consumo)
    for _ in range(20):
        env.step()
        
    post_diffusion_mass = np.sum(env.glucose_grid.grid)
    
    # La masa total debe ser mayor o igual debido al flujo de entrada Dirichlet en el borde izquierdo
    assert post_diffusion_mass >= initial_mass, "La masa disminuyó sin que se aplicara consumo"
    
    # Aplicar consumo localizado
    consume_amount = 0.1
    # Consumir de glucosa en una posición central
    env.consume(x=15.0, y=15.0, amount=consume_amount, field="glucose")
    
    post_consumption_mass = np.sum(env.glucose_grid.grid)
    
    # La masa total debe haber disminuido exactamente tras el consumo
    assert post_consumption_mass < post_diffusion_mass, "La masa no disminuyó después del consumo"
    # Y la diferencia debe ser menor o igual a consume_amount (pude haber sido acotada si el casillero tenía menos)
    assert np.isclose(post_diffusion_mass - post_consumption_mass, consume_amount) or (post_diffusion_mass - post_consumption_mass < consume_amount)

def test_steady_state_convergence():
    """
    Verifica que el campo de difusión converge a un estado estacionario en menos de 5000 pasos sin agentes.
    """
    # Usar una rejilla pequeña para acelerar el test de convergencia
    env = EnvironmentFields(size=(15, 15), dx=1.0, dt=0.01)
    
    converged = False
    tolerance = 1e-6
    max_steps = 5000
    
    for step_idx in range(max_steps):
        prev_grid = env.glucose_grid.grid.copy()
        env.step()
        
        # Calcular el cambio medio absoluto entre pasos sucesivos
        change = np.mean(np.abs(env.glucose_grid.grid - prev_grid))
        
        if change < tolerance:
            converged = True
            print(f"Convergió a estado estacionario en {step_idx} pasos.")
            break
            
    assert converged, f"El campo no converge a un estado estacionario dentro de los {max_steps} pasos establecidos."

def test_rendering_format():
    """
    Verifica que el método render() devuelva un array RGB de dimensiones válidas
    y formato np.uint8 en el rango [0, 255].
    """
    env = EnvironmentFields(size=(40, 40), dx=1.0, dt=0.01)
    
    # Renderizar glucosa y oxígeno
    img_glucose = env.render(field="glucose")
    img_oxygen = env.render(field="oxygen")
    
    # Verificar forma (ny, nx, 3)
    assert img_glucose.shape == (40, 40, 3)
    assert img_oxygen.shape == (40, 40, 3)
    
    # Verificar tipo uint8
    assert img_glucose.dtype == np.uint8
    assert img_oxygen.dtype == np.uint8
    
    # Verificar que los valores estén en el rango de color de 8 bits
    assert img_glucose.min() >= 0
    assert img_glucose.max() <= 255
