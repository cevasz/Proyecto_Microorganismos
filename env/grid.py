import numpy as np
from .diffusion import DiffusionGrid

class EnvironmentFields:
    """
    Coordinador maestro de los campos físicos químicos del entorno de simulación.
    Administra la dinámica paralela de Glucosa y Oxígeno a través de DiffusionGrids,
    calcula gradientes reales y maneja el acoplamiento bidireccional del consumo bacteriano.
    """
    def __init__(self, size=(50, 50), dx=1.0, dt=0.01):
        """
        Args:
            size (tuple): Dimensiones espaciales del plato (ny, nx) en número de celdas.
            dx (float): Tamaño real de cada celda en μm.
            dt (float): Intervalo de tiempo biológico de un paso de simulación.
        """
        self.ny, self.nx = size
        self.dx = dx
        self.dt = dt
        
        # 1. Rejilla de Glucosa (Difusión lenta, D = 600 μm²/s)
        self.glucose_grid = DiffusionGrid(size=size, dx=dx, dt=dt, D=600.0, init_val=0.0, boundary_val=1.0)
        
        # 2. Rejilla de Oxígeno (Difusión rápida, D = 2000 μm²/s)
        self.oxygen_grid = DiffusionGrid(size=size, dx=dx, dt=dt, D=2000.0, init_val=0.0, boundary_val=1.0)
        
        # Centro de atracción del gradiente / fuente de alimento (borde izquierdo, centro vertical)
        self.center = np.array([0.0, (self.ny * self.dx) / 2.0])

    def step(self):
        """
        Avanza la física espacial de ambos campos por un intervalo dt de simulación.
        """
        self.glucose_grid.step()
        self.oxygen_grid.step()

    def _map_coords(self, x, y):
        """
        Mapea coordenadas espaciales reales en μm a índices de la matriz del plato (i, j).
        """
        j = int(np.clip(x / self.dx, 0, self.nx - 1))
        i = int(np.clip(y / self.dx, 0, self.ny - 1))
        return i, j

    def get_observation(self, x, y):
        """
        Retorna la observación del entorno real percibida por una bacteria en (x, y):
        [glucosa, gradiente_g, oxigeno, gradiente_o]
        """
        i, j = self._map_coords(x, y)
        
        glucose = float(self.glucose_grid.grid[i, j])
        oxygen = float(self.oxygen_grid.grid[i, j])
        
        # Calcular vecindad espacial para derivadas de diferencias centrales
        jp = min(j + 1, self.nx - 1)
        jm = max(j - 1, 0)
        ip = min(i + 1, self.ny - 1)
        im = max(i - 1, 0)
        
        # Gradiente real de glucosa
        grad_g_x = (self.glucose_grid.grid[i, jp] - self.glucose_grid.grid[i, jm]) / (2.0 * self.dx if jp != jm else 1.0)
        grad_g_y = (self.glucose_grid.grid[ip, j] - self.glucose_grid.grid[im, j]) / (2.0 * self.dx if ip != im else 1.0)
        grad_g = float(np.sqrt(grad_g_x**2 + grad_g_y**2))
        
        # Gradiente real de oxígeno
        grad_o_x = (self.oxygen_grid.grid[i, jp] - self.oxygen_grid.grid[i, jm]) / (2.0 * self.dx if jp != jm else 1.0)
        grad_o_y = (self.oxygen_grid.grid[ip, j] - self.oxygen_grid.grid[im, j]) / (2.0 * self.dx if ip != im else 1.0)
        grad_o = float(np.sqrt(grad_o_x**2 + grad_o_y**2))
        
        return [glucose, grad_g, oxygen, grad_o]

    def consume(self, x, y, amount, field="glucose"):
        """
        Aplica el consumo localizado de biomasa de una bacteria sobre la celda (x, y),
        modificando directamente la concentración de la rejilla.
        Retorna la cantidad neta consumida (acotada por la disponibilidad física).
        """
        i, j = self._map_coords(x, y)
        grid_obj = self.glucose_grid if field == "glucose" else self.oxygen_grid
        
        available = grid_obj.grid[i, j]
        consumed = min(available, amount)
        
        # Descontar físicamente del casillero químico
        grid_obj.grid[i, j] -= consumed
        return float(consumed)

    def render(self, field="glucose"):
        """
        Genera una textura RGB compatible con Pygame/Matplotlib en formato uint8 [0, 255].
        Glucosa es canalizado en verde, Oxígeno en azul.
        """
        grid_obj = self.glucose_grid if field == "glucose" else self.oxygen_grid
        img = np.zeros((self.ny, self.nx, 3), dtype=np.uint8)
        
        if field == "glucose":
            img[:, :, 1] = (grid_obj.grid * 255).astype(np.uint8)  # Verde
        else:
            img[:, :, 2] = (grid_obj.grid * 255).astype(np.uint8)  # Azul
            
        return img

    @property
    def nutrients(self):
        """Retorna la masa total de glucosa libre en la cuadrícula."""
        return float(np.sum(self.glucose_grid.grid))
