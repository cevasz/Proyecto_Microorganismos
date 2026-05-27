import pygame
import pygame_gui
import numpy as np


class ColonyVisualizer:
    """Visualizador 2D mejorado con menú desplegable de estadísticas.

    * Panel izquierdo: entorno (gradiente de glucosa) y agentes.
    * Panel derecho: gráfica dinámica de la métrica seleccionada.
    * Menú superior (pygame_gui) permite elegir la métrica a graficar.
    """

    def __init__(self, width: int = 1200, height: int = 600, fps: int = 30):
        pygame.init()
        self.width = width
        self.height = height
        self.fps = fps
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("NeuroColony‑EC: Simulación 2D elegante")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)

        # UI manager (pygame_gui)
        self.ui_manager = pygame_gui.UIManager((self.width, self.height))
        self.metric_options = ["Población", "Nutrientes", "Divisiones"]
        self.dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=self.metric_options,
            starting_option=self.metric_options[0],
            relative_rect=pygame.Rect((self.width // 2 + 10, 10), (200, 30)),
            manager=self.ui_manager,
        )

        # Surface for the right‑hand graph
        self.graph_surface = pygame.Surface((self.width // 2 - 20, self.height - 50))
        self.time_history: list[float] = []
        self.metric_history: list[float] = []

    def _draw_gradient(self, center: tuple[int, int], max_radius: int = 200) -> None:
        """Draw a smooth radial green gradient that represents glucose concentration.
        The gradient is built with semi‑transparent concentric circles for a fluid look.
        """
        for radius in range(max_radius, 0, -20):
            intensity = int(255 * (1 - radius / max_radius))
            color = (0, intensity, 0, 150)  # green with alpha
            circle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, color, (radius, radius), radius)
            self.screen.blit(circle_surf, (center[0] - radius, center[1] - radius))

    def _draw_agents(self, colony, scale_x: float, scale_y: float) -> None:
        """Render agents as anti‑aliased blue circles with a subtle shadow."""
        for agent in colony.agents:
            px = int(agent.position[0] * scale_x)
            py = int(agent.position[1] * scale_y)
            if 0 <= px < self.width // 2 and 0 <= py < self.height:
                # shadow
                pygame.draw.circle(self.screen, (0, 0, 0, 80), (px + 2, py + 2), 4)
                # main body
                pygame.draw.circle(self.screen, (100, 200, 255), (px, py), 3)

    def _update_metric_history(self, step: int, dt: float, colony, env_fields) -> None:
        """Append the selected metric value to the history lists.
        The dropdown determines which statistic is tracked.
        """
        selected = self.dropdown.selected_option
        self.time_history.append(step * dt)
        if selected == "Población":
            self.metric_history.append(colony.get_population_stats()["N"])
        elif selected == "Nutrientes":
            # Works for both real and mock environments
            if hasattr(env_fields, "nutrients"):
                self.metric_history.append(float(env_fields.nutrients))
            else:
                self.metric_history.append(0.0)
        elif selected == "Divisiones":
            # Count agents that just divided this step (attribute set by colony)
            divs = sum(1 for a in colony.agents if getattr(a, "just_divided", False))
            self.metric_history.append(divs)
        else:
            self.metric_history.append(0)

    def _draw_graph(self) -> None:
        """Render the time‑series of the selected metric on the right panel."""
        self.graph_surface.fill((250, 250, 250))
        if len(self.metric_history) > 1:
            max_val = max(50, max(self.metric_history) * 1.2)
            points = []
            for i, val in enumerate(self.metric_history):
                gx = int((i / 1000.0) * self.graph_surface.get_width())
                gy = self.graph_surface.get_height() - int((val / max_val) * self.graph_surface.get_height())
                points.append((gx, gy))
            pygame.draw.lines(self.graph_surface, (200, 50, 50), False, points, 2)
        # metric title
        title_surf = self.font.render(self.dropdown.selected_option, True, (0, 0, 0))
        self.graph_surface.blit(title_surf, (10, 10))
        # blit onto main screen
        self.screen.blit(self.graph_surface, (self.width // 2 + 10, 50))

    def draw(self, colony, env_fields, step: int, dt: float) -> None:
        """Main rendering routine called each simulation step.
        Handles events, UI updates, draws the environment, agents and the metric graph.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
            self.ui_manager.process_events(event)

        self.screen.fill((20, 20, 20))

        # LEFT PANEL – environment + agents
        left_rect = pygame.Rect(0, 0, self.width // 2, self.height)
        pygame.draw.rect(self.screen, (25, 25, 30), left_rect)
        center_x = self.width // 4
        center_y = self.height // 2
        self._draw_gradient((center_x, center_y), max_radius=180)
        scale_x = (self.width // 2) / 100.0
        scale_y = self.height / 100.0
        self._draw_agents(colony, scale_x, scale_y)

        # UI (dropdown) – placed on top of left panel
        self.ui_manager.update(0)
        self.ui_manager.draw_ui(self.screen)

        # RIGHT PANEL – real‑time graph
        self._update_metric_history(step, dt, colony, env_fields)
        self._draw_graph()

        # simple HUD
        hud = self.font.render(f"Paso: {step} | Agentes: {colony.get_population_stats()[\"N\"]}", True, (255, 255, 255))
        self.screen.blit(hud, (15, 15))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        pygame.quit()
