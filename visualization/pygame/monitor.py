import pygame
from .colors import COLORS
from .chart_data import ChartData
from .grid_renderer import draw_grid
from .chart_renderer import update_chart_data, draw_charts
from .ui_renderer import (
    draw_legend,
    draw_buttons,
    draw_status,
    draw_live_counters,
    draw_corner_overlay,
)
from .event_handler import handle_events


class PygameMonitor:
    def __init__(self, sim, cfg):
        self.sim = sim
        self.cfg = cfg

        # --- Initialize pygame ---
        pygame.init()
        self.width, self.height = 1920, 1000
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Civilization Simulation - Pygame Monitor")

        # Fonts
        self.fonts = {
            "small": pygame.font.Font(None, 18),
            "medium": pygame.font.Font(None, 22),
            "large": pygame.font.Font(None, 28),
            "title": pygame.font.Font(None, 32),
        }

        # Grid layout
        self.grid_size = 1200
        self.grid_x, self.grid_y = 50, 50
        self.cell_size = min(self.grid_size // self.cfg.W, self.grid_size // self.cfg.H)

        # UI state
        self.is_paused = False
        self.should_stop = False
        self.mouse_pos = (0, 0)

        # Chart data
        self.charts = self._init_charts()

        # Buttons & legend
        self.buttons = self._init_buttons()
        self.legend = self._init_legend()

        # FPS control
        self.fps_clock = pygame.time.Clock()
        self.fps = 60

    # ---------- Initialization helpers ----------

    def _init_charts(self):
        return {
            "population": ChartData([], 0, 0, (31, 119, 180), "Population"),
            "energy": ChartData([], 0, 0, (44, 160, 44), "Mean Energy"),
            "strength_dist": ChartData([], 0, 0, (255, 127, 0), "Strength Distribution"),
            "perception_dist": ChartData([], 0, 0, (200, 20, 0), "Perception Distribution"),
            "aggression_dist": ChartData([], 0, 0, (200, 20, 0), "Aggression Distribution"),
            "daily_stats": ChartData([], 0, 0, (128, 0, 128), "Daily Statistics"),
        }

    def _init_buttons(self):
        button_width, button_height = 120, 40
        button_y = self.height - 60
        return {
            "pause_play": {
                "rect": pygame.Rect(self.width // 2 - button_width - 20, button_y, button_width, button_height),
                "text": "⏸ Pause",
                "color": COLORS["UI_BUTTON"],
                "hover_color": COLORS["UI_BUTTON_HOVER"],
                "action": "toggle_pause",
            },
            "stop": {
                "rect": pygame.Rect(self.width // 2 + 20, button_y, button_width, button_height),
                "text": "⏹ Stop",
                "color": COLORS["UI_STOP"],
                "hover_color": (255, 150, 150),
                "action": "stop",
            },
        }

    def _init_legend(self):
        legend_x, legend_y, legend_width = self.grid_x + self.grid_size + 20, 50, 250
        items = [
            {"type": "zone", "color": COLORS["ZONE_LOW"], "label": "Low Food Zone"},
            {"type": "zone", "color": COLORS["ZONE_MID"], "label": "Medium Food Zone"},
            {"type": "zone", "color": COLORS["ZONE_HIGH"], "label": "High Food Zone"},
            {"type": "carrot", "color": COLORS["CARROT"], "label": "Carrots"},
            {"type": "energy", "color": COLORS["ENERGY_LOW"], "label": "Low Energy"},
            {"type": "energy", "color": COLORS["ENERGY_MID"], "label": "Medium Energy"},
            {"type": "energy", "color": COLORS["ENERGY_HIGH"], "label": "High Energy"},
            {"type": "action", "color": COLORS["ACTION_SEEK"], "label": "Seeking Food"},
            {"type": "action", "color": COLORS["ACTION_EXPLORE"], "label": "Exploring"},
            {"type": "action", "color": COLORS["ACTION_FORAGE"], "label": "Foraging"},
            {"type": "action", "color": COLORS["ACTION_COMBAT"], "label": "In Combat"},
        ]
        return {"x": legend_x, "y": legend_y, "width": legend_width, "items": items}

    # ---------- Main loop ----------

    def render(self, day_idx, tick_idx):
        """Render one frame per simulation tick"""
        if not handle_events(self):
            return False

        update_chart_data(self)

        self.screen.fill(COLORS["UI_BACKGROUND"])
        draw_grid(self)
        draw_legend(self)
        draw_charts(self)
        draw_buttons(self)
        draw_status(self)
        draw_live_counters(self)
        draw_corner_overlay(self)

        pygame.display.flip()
        self.fps_clock.tick(self.fps)
        return True

    # ---------- Utility ----------

    def should_continue(self):
        return not self.should_stop

    def cleanup(self):
        pygame.quit()
