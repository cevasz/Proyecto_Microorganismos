import numpy as np

class DiffusionGrid:
    """
    Solucionador numérico en 2D de la ecuación de difusión utilizando el método
    de diferencias finitas explícitas (FDM).
    Implementa sub-stepping dinámico basado en la condición de estabilidad CFL
    (Courant-Friedrichs-Lewy) para evitar divergencia y NaNs.
    """
    def __init__(self, size=(50, 50), dx=1.0, dt=0.01, D=600.0, init_val=0.0, boundary_val=1.0):
        """
        Args:
            size (tuple): Dimensiones de la rejilla (ny, nx).
            dx (float): Resolución espacial de celda en μm.
            dt (float): Paso temporal global de la simulación en segundos.
            D (float): Coeficiente de difusión en μm²/s.
            init_val (float): Concentración inicial en el interior de la rejilla.
            boundary_val (float): Valor de frontera Dirichlet constante en la columna izquierda.
        """
        self.ny, self.nx = size
        self.dx = dx
        self.dt = dt
        self.D = D
        self.boundary_val = boundary_val
        
        # Inicialización de la rejilla de concentraciones
        self.grid = np.full((self.ny, self.nx), init_val, dtype=np.float32)
        # Establecer la condición Dirichlet inicial en el borde izquierdo (columna 0)
        self.grid[:, 0] = self.boundary_val
        
        # Cálculo dinámico de sub-stepping CFL para asegurar estabilidad numérica
        # En 2D, para diferencias explícitas ordinarias, dt <= dx^2 / (4 * D)
        dt_stable_limit = (self.dx ** 2) / (4.0 * self.D)
        # Usar un factor de seguridad del 10%
        self.dt_stable = dt_stable_limit * 0.9
        
        if self.dt > self.dt_stable:
            self.n_substeps = int(np.ceil(self.dt / self.dt_stable))
            self.dt_sub = self.dt / self.n_substeps
        else:
            self.n_substeps = 1
            self.dt_sub = self.dt

    def step(self):
        """
        Avanza la ecuación de difusión por un intervalo dt completo de simulación,
        ejecutando internamente n_substeps estables explícitos.
        """
        for _ in range(self.n_substeps):
            grid_new = self.grid.copy()
            
            # 1. Calcular el Laplaciano en 2D en los nodos interiores (NumPy vectorizado)
            # nodes interiors: [1:-1, 1:-1]
            laplacian = (
                self.grid[2:, 1:-1] + self.grid[:-2, 1:-1] +
                self.grid[1:-1, 2:] + self.grid[1:-1, :-2] -
                4.0 * self.grid[1:-1, 1:-1]
            )
            
            # 2. Actualizar concentraciones interiores usando Euler explícito
            grid_new[1:-1, 1:-1] = self.grid[1:-1, 1:-1] + self.D * (self.dt_sub / (self.dx ** 2)) * laplacian
            
            # 3. Aplicar condiciones de frontera
            # Borde Izquierdo (X=0, columna 0) -> Dirichlet: concentración constante inyectada
            grid_new[:, 0] = self.boundary_val
            
            # Borde Derecho (X=nx-1) -> Neumann: flujo libre nulo (pared impermeable)
            grid_new[:, -1] = grid_new[:, -2]
            
            # Borde Superior (Y=0) -> Neumann: flujo libre nulo
            grid_new[0, :] = grid_new[1, :]
            
            # Borde Inferior (Y=ny-1) -> Neumann: flujo libre nulo
            grid_new[-1, :] = grid_new[-2, :]
            
            # 4. Asegurar no-negatividad física e integridad numérica [0.0, 1.0]
            self.grid = np.clip(grid_new, 0.0, 1.0)
