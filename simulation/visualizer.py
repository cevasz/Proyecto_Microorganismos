import pygame
import numpy as np

class ColonyVisualizer:
    """
    Visualizador en tiempo real basado en Pygame.
    Panel Izquierdo: Entorno espacial (Glucosa y Agentes)
    Panel Derecho: Gráfica poblacional de supervivencia
    """
    def __init__(self, width=1200, height=600, fps=30):
        pygame.init()
        self.width = width
        self.height = height
        self.fps = fps
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("NeuroColony-EC: Simulación Quimiotáctica 2D")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        
        # Superficie independiente para la gráfica derecha
        self.graph_surface = pygame.Surface((self.width // 2, self.height))
        self.time_history = []
        self.pop_history = []
        
    def draw(self, colony, env_fields, step, dt):
        """
        Dibuja los paneles en cada step requerido.
        Maneja subsampling si se le llama con intervalos para proteger los FPS.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
                
        self.screen.fill((20, 20, 20)) 
        
        # ==========================================
        # PANEL IZQUIERDO (Espacio 2D)
        # ==========================================
        left_rect = pygame.Rect(0, 0, self.width // 2, self.height)
        pygame.draw.rect(self.screen, (25, 25, 30), left_rect)
        
        # Simulación visual del gradiente (fuente en el centro del plano 100x100)
        center_x = int((self.width // 2) * 0.5)
        center_y = int(self.height * 0.5)
        for radius in range(200, 0, -20):
            intensity = int(255 * (1 - radius/200.0))
            pygame.draw.circle(self.screen, (0, intensity, 0), (center_x, center_y), radius, 2)
        
        # Conversión de coordenadas biológicas (100x100 μm) a píxeles
        scale_x = (self.width // 2) / 100.0
        scale_y = self.height / 100.0
        
        # Dibujar Agentes E. coli (puntos azules)
        for agent in colony.agents:
            px = int(agent.position[0] * scale_x)
            py = int(agent.position[1] * scale_y)
            # Evitar dibujar fuera de panel
            if px < self.width//2 and py < self.height:
                pygame.draw.circle(self.screen, (100, 200, 255), (px, py), 3)
            
        # ==========================================
        # PANEL DERECHO (Estadísticas y Gráfica)
        # ==========================================
        self.graph_surface.fill((240, 240, 240))
        stats = colony.get_population_stats()
        N = stats["N"]
        
        self.time_history.append(step * dt)
        self.pop_history.append(N)
        
        # Ventana deslizante para la gráfica (últimos 1000 registros)
        if len(self.time_history) > 1000:
            self.time_history.pop(0)
            self.pop_history.pop(0)
            
        if len(self.pop_history) > 1:
            max_pop = max(50, max(self.pop_history) * 1.2) # Margen dinámico
            points = []
            for i, pop in enumerate(self.pop_history):
                gx = int((i / 1000.0) * (self.width // 2))
                gy = self.height - int((pop / max_pop) * self.height)
                points.append((gx, gy))
            pygame.draw.lines(self.graph_surface, (200, 50, 50), False, points, 2)
            
        self.screen.blit(self.graph_surface, (self.width // 2, 0))
        
        # Texto HUD
        hud_text = self.font.render(f"Población: {N} | Paso: {step}", True, (255, 255, 255))
        self.screen.blit(hud_text, (15, 15))
        
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
