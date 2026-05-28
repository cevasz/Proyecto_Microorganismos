import pygame
import numpy as np
import time
from collections import deque

class ColonyVisualizer:
    """
    Visualizador 2D Premium para NeuroColony-EC (Edición MVP).
    
    * Interfaz unificada de 1200x700 píxeles.
    * Panel Izquierdo (Cuadrado de 700x700): Simulación con fondo de cuadrícula,
      mapa de calor FDM real de glucosa (PDE), y bacterias con trails y dirección.
    * Panel Derecho (500x700): HUD glassmorphic con explicación del sistema,
      KPIs en tiempo real, y gráfica lineal interactiva en vanilla Pygame.
    * 100% libre de dependencias externas inestables (sin pygame_gui).
    """
    def __init__(self, width: int = 1200, height: int = 700, fps: int = 30):
        pygame.init()
        self.width = width
        self.height = height
        self.fps = fps
        
        # Inicializar pantalla principal
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("NeuroColony-EC | Simulación Multiagente de E. coli")
        self.clock = pygame.time.Clock()
        
        # Configurar fuentes de sistema premium
        try:
            self.font_title = pygame.font.SysFont("Segoe UI", 24, bold=True)
            self.font_section = pygame.font.SysFont("Segoe UI", 16, bold=True)
            self.font_body = pygame.font.SysFont("Segoe UI", 13)
            self.font_hud_lbl = pygame.font.SysFont("Segoe UI", 11, bold=True)
            self.font_hud_val = pygame.font.SysFont("Segoe UI", 20, bold=True)
        except Exception:
            # Fallback en caso de que no existan las fuentes de Segoe
            self.font_title = pygame.font.Font(None, 32)
            self.font_section = pygame.font.Font(None, 24)
            self.font_body = pygame.font.Font(None, 18)
            self.font_hud_lbl = pygame.font.Font(None, 16)
            self.font_hud_val = pygame.font.Font(None, 28)
            
        # Paleta de Colores Sci-Fi / Bio-Tech Premium
        # Paleta de Colores Sci-Fi / Bio-Tech Premium Softer Matte
        self.COLOR_BG = (30, 33, 41)          # Carbón mate suave y sofisticado
        self.COLOR_PANEL_BG = (38, 42, 53)     # Gris traslúcido suave
        self.COLOR_PANEL_BORDER = (60, 66, 82) # Gris azulado medio elegante
        self.COLOR_GRID = (38, 42, 53)         # Cuadrícula muy sutil y agradable
        self.COLOR_TEXT_PR = (248, 250, 252)   # Blanco roto primario
        self.COLOR_TEXT_SEC = (148, 163, 184) # Gris secundario
        self.COLOR_ACCENT = (74, 197, 253)     # Cian bio-luminoso suave
        
        self.COLOR_GLUCOSE = (251, 191, 36)    # Dorado ámbar premium para nutrientes
        self.COLOR_OXYGEN = (56, 189, 248)     # Turquesa suave para oxígeno
        
        # Estados dinámicos de bacterias según energía
        self.COLOR_ENERGY_HIGH = (34, 211, 238) # Cian suave
        self.COLOR_ENERGY_MED = (251, 191, 36) # Naranja/Ámbar suave
        self.COLOR_ENERGY_LOW = (251, 113, 133)  # Rosa/Coral suave (Inanición)
        
        # Historial de métricas para la gráfica interactiva
        self.time_history = []
        self.population_history = []
        self.nutrients_history = []
        self.divisions_history = []
        
        # Selección de métrica activa (0 = Población, 1 = Nutrientes, 2 = Divisiones)
        self.selected_metric = 0
        self.total_cumulative_divisions = 0
        
        # Para controlar eventos de mouse sin repeticiones de clics
        self.last_mouse_state = False

    def draw_card(self, surface, rect, title):
        """Dibuja una tarjeta con estilo glassmorphic moderno y un sutil borde elegante."""
        # Dibujar fondo de tarjeta
        pygame.draw.rect(surface, self.COLOR_PANEL_BG, rect, border_radius=8)
        # Dibujar borde fino
        pygame.draw.rect(surface, self.COLOR_PANEL_BORDER, rect, width=1, border_radius=8)
        # Dibujar cabecera
        if title:
            lbl_surf = self.font_section.render(title, True, self.COLOR_ACCENT)
            surface.blit(lbl_surf, (rect.x + 15, rect.y + 12))

    def draw_wrapped_text(self, surface, text, x, y, max_width, font, color):
        """Función auxiliar que ajusta y dibuja un texto multilínea sin desbordar el contenedor."""
        words = text.split(' ')
        lines = []
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            if font.size(test_line)[0] < max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
            
        curr_y = y
        for line in lines:
            txt_surf = font.render(line, True, color)
            surface.blit(txt_surf, (x, curr_y))
            curr_y += font.get_linesize() + 3
        return curr_y

    def _draw_grid(self, surface, size=700, spacing=35):
        """Dibuja la cuadrícula de simulación matemática para dar profundidad visual."""
        for x in range(0, size, spacing):
            pygame.draw.line(surface, self.COLOR_GRID, (x, 0), (x, size), 1)
        for y in range(0, size, spacing):
            pygame.draw.line(surface, self.COLOR_GRID, (0, y), (size, y), 1)

    def _draw_food_heatmap(self, surface, env_fields, size=700):
        """
        Dibuja los nutrientes como un campo de micropartículas doradas/ámbar flotantes 
        y resplandecientes en lugar de un mapa de calor verde de fondo, eliminando el color verde
        y creando una estética premium de laboratorio de biotecnología.
        """
        # Extraer u obtener la cuadrícula de nutrientes en resolución 100x100
        if hasattr(env_fields, "glucose_grid"):
            grid = env_fields.glucose_grid.grid
        else:
            # Fallback robusto para MockEnvFields
            ny, nx = 100, 100
            y_idx, x_idx = np.indices((ny, nx))
            # Calcular distancia al centro en celdas (center es 50,50 en mock)
            cx, cy = env_fields.center[0], env_fields.center[1]
            dist = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
            nut_factor = min(1.0, env_fields.nutrients / 5000.0)
            grid = np.clip(1.0 - (dist / 40.0), 0.0, 1.0) * nut_factor
            
        ny, nx = grid.shape
        scale = size / 100.0
        
        # Crear superficie para transparencias de nutrientes
        nutrient_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Submuestreo para evitar sobrecarga en la GPU/CPU y dibujar un campo de partículas elegante
        step_x = 2
        step_y = 2
        
        for y in range(0, ny, step_y):
            for x in range(0, nx, step_x):
                val = grid[y, x]
                if val > 0.02:
                    # Calcular posición en pantalla
                    px = int(x * scale)
                    py = int(y * scale)
                    
                    # El tamaño y opacidad dependen de la concentración de nutrientes
                    radius = int(1 + val * 4)
                    alpha = int(40 + val * 200)
                    
                    # Dibujar partícula dorada/ámbar (253, 186, 116)
                    pygame.draw.circle(nutrient_surface, (253, 186, 116, alpha), (px, py), radius)
                    if val > 0.4:
                        # Resplandor sutil (glow) para acumulaciones densas
                        pygame.draw.circle(nutrient_surface, (253, 186, 116, int(alpha * 0.25)), (px, py), radius * 2)
                        
        surface.blit(nutrient_surface, (0, 0))

    def _draw_agents_and_trails(self, surface, colony, size=700):
        """
        Dibuja los caminos recorridos y a los propios agentes bacterianos como hermosas cápsulas
        alargadas (rod-shaped) de E. coli con un núcleo de proteínas interno brillante,
        un suave resplandor exterior y sus estelas de movimiento.
        """
        # Crear superficie para canal alfa de trails y resplandores (glows)
        overlay = pygame.Surface((size, size), pygame.SRCALPHA)
        
        scale = size / 100.0 # Posiciones están en escala [0, 100] micrómetros
        
        for agent in colony.agents:
            # 1. Dibujar estela de movimiento (trails) con opacidad degradada
            if len(agent.pos_history) > 1:
                hist_len = len(agent.pos_history)
                for idx in range(hist_len - 1):
                    p1 = agent.pos_history[idx]
                    p2 = agent.pos_history[idx+1]
                    
                    x1, y1 = int(p1[0] * scale), int(p1[1] * scale)
                    x2, y2 = int(p2[0] * scale), int(p2[1] * scale)
                    
                    # Alpha progresivo: los puntos antiguos son casi invisibles, los recientes brillan más
                    alpha = int(140 * (idx / hist_len))
                    pygame.draw.line(overlay, (56, 189, 248, alpha), (x1, y1), (x2, y2), 2)
            
            # Obtener datos cinemáticos
            ax, ay = int(agent.position[0] * scale), int(agent.position[1] * scale)
            energy = getattr(agent, "energy", 1.0)
            
            # 2. Elegir color dinámico según energía
            if energy > 0.8:
                core_color = self.COLOR_ENERGY_HIGH
            elif energy > 0.4:
                core_color = self.COLOR_ENERGY_MED
            else:
                core_color = self.COLOR_ENERGY_LOW
                
            # 3. Dibujar halo luminoso (Glow) en el overlay con transparencia
            pygame.draw.circle(overlay, (*core_color, 35), (ax, ay), 14)
            
            # 4. Dibujar cuerpo bacteriano como cápsula elegante (rod-shaped E. coli) en la pantalla principal
            if 0 <= ax < size and 0 <= ay < size:
                # Longitud física y ancho de la bacteria en pantalla
                bact_len = max(6.0, agent.length * 8.0)
                bact_width = 8
                
                cos_o = np.cos(agent.orientation)
                sin_o = np.sin(agent.orientation)
                
                # Extremos frontales y traseros para dibujar la cápsula
                x_front = ax + int(bact_len * cos_o)
                y_front = ay + int(bact_len * sin_o)
                x_back = ax - int(bact_len * cos_o)
                y_back = ay - int(bact_len * sin_o)
                
                # A. Dibujar el cuerpo principal y membrana (capsula exterior de color de energía)
                pygame.draw.line(surface, core_color, (x_back, y_back), (x_front, y_front), bact_width)
                pygame.draw.circle(surface, core_color, (x_front, y_front), bact_width // 2)
                pygame.draw.circle(surface, core_color, (x_back, y_back), bact_width // 2)
                
                # B. Dibujar el núcleo de proteínas brillante interno (citoplasma / nucleoide visible)
                inner_color = (255, 255, 255)  # Núcleo de proteínas brillante
                inner_width = bact_width - 4
                inner_len = bact_len - 2
                
                if inner_width >= 2 and inner_len > 0:
                    ix_front = ax + int(inner_len * cos_o)
                    iy_front = ay + int(inner_len * sin_o)
                    ix_back = ax - int(inner_len * cos_o)
                    iy_back = ay - int(inner_len * sin_o)
                    
                    pygame.draw.line(surface, inner_color, (ix_back, iy_back), (ix_front, iy_front), inner_width)
                    pygame.draw.circle(surface, inner_color, (ix_front, iy_front), inner_width // 2)
                    pygame.draw.circle(surface, inner_color, (ix_back, iy_back), inner_width // 2)
                
                # 5. Dibujar indicador sutil de dirección (Vector "nariz" blanco)
                nose_len = bact_len + 4
                nx = ax + int(cos_o * nose_len)
                ny = ay + int(sin_o * nose_len)
                pygame.draw.line(surface, (255, 255, 255), (x_front, y_front), (nx, ny), 1)

        # Proyectar el overlay de transparencia sobre el cuadrante de simulación
        surface.blit(overlay, (0, 0))

    def _draw_food_emitter(self, surface, env_fields, size=700):
        """Dibuja un emisor de alimento pulsante e hiper-claro en el centro/borde de recursos."""
        scale = size / 100.0
        cx = int(env_fields.center[0] * scale)
        cy = int(env_fields.center[1] * scale)
        
        # Pulso radial de ondas
        t = time.time()
        for ring in range(3):
            pulse_factor = ((t * 2.0 + ring * 0.3) % 1.0)
            radius = int(5 + pulse_factor * 35)
            alpha = int(180 * (1.0 - pulse_factor))
            
            # Anillo de onda expansiva dorada cálida (coherente con las micropartículas de glucosa)
            pulse_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(pulse_surf, (253, 186, 116, alpha), (radius, radius), radius, 2)
            surface.blit(pulse_surf, (cx - radius, cy - radius))
            
        # Faro central luminoso
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 6)
        pygame.draw.circle(surface, (251, 191, 36), (cx, cy), 4)

    def update_stats(self, step, dt, colony, env_fields):
        """Actualiza y registra los buffers de métricas en tiempo real de forma continua en cada paso de simulación."""
        N = len(colony.agents)
        
        # 1. Totalizar divisiones ocurridas en el paso actual (bacterias nacidas con edad 0 en este paso)
        # Solo contamos si step > 0 para evitar contar la población inicial como nuevas divisiones
        divs_step = 0
        if step > 0:
            divs_step = sum(1 for a in colony.agents if getattr(a, "age", 0) == 0)
        self.total_cumulative_divisions += divs_step
        
        # 2. Registrar concentraciones globales
        if hasattr(env_fields, "nutrients"):
            nutrients = float(env_fields.nutrients)
        else:
            # Obtener masa total en el mock
            nutrients = 0.0
            
        self.time_history.append(step * dt)
        self.population_history.append(N)
        self.nutrients_history.append(nutrients)
        self.divisions_history.append(self.total_cumulative_divisions)
        
        # Recortar el historial si es demasiado largo para evitar lentitud (máx 500 puntos)
        max_points = 500
        if len(self.time_history) > max_points:
            self.time_history.pop(0)
            self.population_history.pop(0)
            self.nutrients_history.pop(0)
            self.divisions_history.pop(0)

    def _draw_side_hud(self, colony, env_fields, step, dt):
        """Renderiza todo el panel de control, explicación y estadísticas del MVP."""
        panel_x = 715
        panel_width = 470
        
        # Fondo general del panel derecho
        pygame.draw.rect(self.screen, (30, 33, 41), (700, 0, 500, self.height))
        pygame.draw.line(self.screen, self.COLOR_PANEL_BORDER, (700, 0), (700, self.height), 2)
        
        # -----------------------------------------------------------------
        # CARD 1: Cabecera y Título Científico
        # -----------------------------------------------------------------
        self.screen.blit(self.font_title.render("NeuroColony-EC", True, self.COLOR_TEXT_PR), (panel_x + 5, 20))
        self.screen.blit(self.font_body.render("Simulación Multiagente de Quimiotaxis de E. coli", True, self.COLOR_ACCENT), (panel_x + 5, 52))
        pygame.draw.line(self.screen, self.COLOR_PANEL_BORDER, (panel_x, 75), (panel_x + panel_width, 75), 1)

        # -----------------------------------------------------------------
        # CARD 2: Explicación de la Simulación (PPO + NODE + LSTM)
        # -----------------------------------------------------------------
        card_explanation_rect = pygame.Rect(panel_x, 90, panel_width, 175)
        self.draw_card(self.screen, card_explanation_rect, "Fundamento de Inteligencia Artificial")
        
        explanation_text = (
            "Esta simulación modela la respuesta biológica celular real. "
            "1. Quimiotaxis RL (Módulo 1): El agente bacteria percibe gradientes químicos y utiliza "
            "una red Actor-Crítico PPO para decidir duración de corridas o giros estocásticos. "
            "2. Señalización PI-NODE (Módulo 2): La transducción del receptor a la proteína CheY-P se simula "
            "con Ecuaciones Diferenciales Ordinarias Neuronales restringidas físicamente. "
            "3. División LSTM (Módulo 3): Una red LSTM evalúa la historia metabólica para inducir fisión biológica."
        )
        self.draw_wrapped_text(
            self.screen, explanation_text, 
            panel_x + 15, 125, panel_width - 30, 
            self.font_body, self.COLOR_TEXT_SEC
        )

        # -----------------------------------------------------------------
        # CARD 3: KPIs e Indicadores Numéricos
        # -----------------------------------------------------------------
        card_kpi_rect = pygame.Rect(panel_x, 280, panel_width, 105)
        self.draw_card(self.screen, card_kpi_rect, "Estado Micro-Ambiental y Población")
        
        # Calcular valores macro
        N = len(colony.agents)
        avg_energy = float(np.mean([a.energy for a in colony.agents])) if N > 0 else 0.0
        elapsed_sec = step * dt
        
        # Dibujar 4 columnas de estadísticas
        col_w = panel_width // 4
        stats = [
            ("POBLACIÓN", f"{N:03d}", self.COLOR_ACCENT),
            ("ENERGÍA", f"{avg_energy * 100:.0f}%", self.COLOR_ENERGY_HIGH if avg_energy > 0.5 else self.COLOR_ENERGY_LOW),
            ("TIEMPO SIM", f"{elapsed_sec:.1f}s", self.COLOR_TEXT_PR),
            ("DIVISIONES", f"{self.total_cumulative_divisions}", self.COLOR_GLUCOSE)
        ]
        
        for idx, (label, val, val_color) in enumerate(stats):
            cx = panel_x + idx * col_w
            # Etiquetas
            lbl_surf = self.font_hud_lbl.render(label, True, self.COLOR_TEXT_SEC)
            self.screen.blit(lbl_surf, (cx + 12, 315))
            # Valores
            val_surf = self.font_hud_val.render(val, True, val_color)
            self.screen.blit(val_surf, (cx + 12, 335))

        # -----------------------------------------------------------------
        # CARD 4: Gráfica Dinámica en Tiempo Real
        # -----------------------------------------------------------------
        card_graph_rect = pygame.Rect(panel_x, 400, panel_width, 275)
        self.draw_card(self.screen, card_graph_rect, "Análisis de Trayectoria Temporal")
        
        # Dibujar marco e interior de la gráfica
        graph_rect = pygame.Rect(panel_x + 15, 435, panel_width - 30, 180)
        pygame.draw.rect(self.screen, (24, 26, 32), graph_rect)
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BORDER, graph_rect, width=1)
        
        # Ejes y líneas de cuadrícula internas de la gráfica
        for gy in range(graph_rect.y + 36, graph_rect.bottom, 36):
            pygame.draw.line(self.screen, (38, 42, 53), (graph_rect.x, gy), (graph_rect.right - 1, gy), 1)
        for gx in range(graph_rect.x + 88, graph_rect.right, 88):
            pygame.draw.line(self.screen, (38, 42, 53), (gx, graph_rect.y), (gx, graph_rect.bottom - 1), 1)
            
        # Determinar buffer a graficar
        if self.selected_metric == 0:
            hist_buffer = self.population_history
            line_color = self.COLOR_ACCENT
            metric_title = "Población (N agentes)"
        elif self.selected_metric == 1:
            hist_buffer = self.nutrients_history
            line_color = self.COLOR_GLUCOSE
            metric_title = "Masa de Glucosa Libre"
        else:
            hist_buffer = self.divisions_history
            line_color = self.COLOR_ENERGY_MED
            metric_title = "Fisiones Acumuladas"
            
        # Dibujar la línea de tendencia de datos
        points_len = len(hist_buffer)
        if points_len > 1:
            max_val = max(10.0, max(hist_buffer) * 1.15)
            min_val = 0.0
            val_range = max_val - min_val
            
            pts = []
            for i, val in enumerate(hist_buffer):
                # Coordenadas X escaladas horizontalmente
                px = graph_rect.x + int((i / (points_len - 1)) * graph_rect.width)
                # Coordenadas Y escaladas verticalmente (invertido en Pygame)
                normalized_y = (val - min_val) / val_range
                py = graph_rect.bottom - int(normalized_y * graph_rect.height)
                pts.append((px, py))
                
            # Renderizar línea de datos suavizada
            pygame.draw.lines(self.screen, line_color, False, pts, 2)
            
            # Dibujar un sutil degradado debajo de la línea
            poly_points = [(graph_rect.x, graph_rect.bottom)] + pts + [(graph_rect.right, graph_rect.bottom)]
            poly_overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(poly_overlay, (*line_color, 25), poly_points)
            self.screen.blit(poly_overlay, (0, 0))
            
            # Mostrar etiquetas de máximo y mínimo alineadas premium en el extremo derecho para no solapar
            lbl_max = self.font_hud_lbl.render(f"Max: {max(hist_buffer):.1f}", True, self.COLOR_TEXT_SEC)
            self.screen.blit(lbl_max, (graph_rect.right - 90, graph_rect.y + 6))
            
            lbl_min = self.font_hud_lbl.render(f"Min: {min(hist_buffer):.1f}", True, self.COLOR_TEXT_SEC)
            self.screen.blit(lbl_min, (graph_rect.right - 90, graph_rect.bottom - 18))
            
        # Texto de la métrica graficada alineado a la izquierda (libre de colisiones)
        lbl_metric = self.font_section.render(metric_title, True, self.COLOR_TEXT_PR)
        self.screen.blit(lbl_metric, (graph_rect.x + 10, graph_rect.y + 6))

        # -----------------------------------------------------------------
        # BOTONES / PESTAÑAS INTERACTIVAS (Vanilla Pygame)
        # -----------------------------------------------------------------
        btn_y = 630
        btn_h = 30
        btn_w = 130
        gap = 15
        
        # Definición de botones coordenados
        self.buttons = [
            {"rect": pygame.Rect(panel_x + 15, btn_y, btn_w, btn_h), "label": "POBLACIÓN", "id": 0, "color": self.COLOR_ACCENT},
            {"rect": pygame.Rect(panel_x + 15 + btn_w + gap, btn_y, btn_w, btn_h), "label": "NUTRIENTES", "id": 1, "color": self.COLOR_GLUCOSE},
            {"rect": pygame.Rect(panel_x + 15 + (btn_w + gap) * 2, btn_y, btn_w, btn_h), "label": "DIVISIONES", "id": 2, "color": self.COLOR_ENERGY_MED}
        ]
        
        # Procesar clics del ratón de forma limpia e interactiva
        mouse_pressed = pygame.mouse.get_pressed()[0]
        mx, my = pygame.mouse.get_pos()
        
        for btn in self.buttons:
            rect = btn["rect"]
            is_hover = rect.collidepoint(mx, my)
            is_active = self.selected_metric == btn["id"]
            
            # Colores dinámicos para feedback táctil
            if is_active:
                bg_color = (*btn["color"], 60)
                border_color = btn["color"]
                txt_color = self.COLOR_TEXT_PR
            elif is_hover:
                bg_color = (45, 55, 72, 120)
                border_color = self.COLOR_ACCENT
                txt_color = self.COLOR_TEXT_PR
            else:
                bg_color = (28, 33, 46, 0)
                border_color = self.COLOR_PANEL_BORDER
                txt_color = self.COLOR_TEXT_SEC
                
            # Dibujar el botón
            btn_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            pygame.draw.rect(btn_surf, bg_color, (0, 0, rect.width, rect.height), border_radius=4)
            pygame.draw.rect(btn_surf, border_color, (0, 0, rect.width, rect.height), width=1, border_radius=4)
            self.screen.blit(btn_surf, (rect.x, rect.y))
            
            # Texto centrado del botón
            lbl_btn = self.font_hud_lbl.render(btn["label"], True, txt_color)
            tx = rect.x + (rect.width - lbl_btn.get_width()) // 2
            ty = rect.y + (rect.height - lbl_btn.get_height()) // 2
            self.screen.blit(lbl_btn, (tx, ty))
            
            # Registrar activación si se hace clic y el estado del mouse cambió
            if is_hover and mouse_pressed and not self.last_mouse_state:
                self.selected_metric = btn["id"]

        self.last_mouse_state = mouse_pressed

    def draw(self, colony, env_fields, step: int, dt: float) -> None:
        """
        Bucle de pintado principal invocado paso a paso en la simulación.
        Actualiza y sincroniza todos los componentes estéticos.
        """
        # 1. Escuchar eventos estándar de salida de Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
                
        # 2. Rellenar pantalla principal con fondo oscuro
        self.screen.fill(self.COLOR_BG)
        
        # 3. Crear el cuadrante de simulación izquierdo (Surface de 700x700)
        sim_surface = pygame.Surface((700, 700))
        sim_surface.fill((24, 26, 32)) # Fondo del plato ligeramente diferenciado (carbón suave)
        
        # 4. Dibujar la cuadrícula en el plato
        self._draw_grid(sim_surface)
        
        # 5. Proyectar mapa de calor FDM real sobre el plato
        self._draw_food_heatmap(sim_surface, env_fields)
        
        # 6. Dibujar la onda expansiva del emisor de comida
        self._draw_food_emitter(sim_surface, env_fields)
        
        # 7. Dibujar bacterias, sus estelas y vectores direccionales
        self._draw_agents_and_trails(sim_surface, colony)
        
        # 8. Blit del plato a la pantalla principal en la izquierda (0,0)
        self.screen.blit(sim_surface, (0, 0))
        
        # 9. Dibujar el panel de control y explicaciones en la derecha (700,0)
        self._draw_side_hud(colony, env_fields, step, dt)
        
        # 11. Flip de Pygame y control de framerate estable
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        """Cierra el subsistema de video Pygame con seguridad."""
        pygame.quit()
