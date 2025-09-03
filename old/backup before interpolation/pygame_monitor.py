# pygame_monitor.py
# A pygame-based live monitor for the civilization simulation
# Replaces matplotlib with better performance and aesthetics

import pygame
import pygame.gfxdraw
import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Color scheme (matching the original)
COLORS = {
    # Food zones (richness levels)
    'ZONE_LOW': (39, 64, 41),      # #274029
    'ZONE_MID': (78, 125, 67),     # #4E7D43  
    'ZONE_HIGH': (155, 208, 109),  # #9BD06D
    
    # Carrots
    'CARROT': (255, 165, 0),       # Orange
    
    # Agent energy levels (viridis-like)
    'ENERGY_LOW': (68, 1, 84),     # Dark purple
    'ENERGY_MID': (59, 82, 139),   # Blue
    'ENERGY_HIGH': (253, 231, 37), # Yellow
    
    # Agent actions (outline colors)
    'ACTION_SEEK': (31, 119, 180),   # Blue
    'ACTION_EXPLORE': (127, 127, 127), # Gray
    'ACTION_FORAGE': (44, 160, 44),   # Green
    'ACTION_COMBAT': (214, 39, 40),   # Red
    
    # UI elements
    'UI_BACKGROUND': (245, 245, 245), # Light gray
    'UI_BORDER': (200, 200, 200),     # Medium gray
    'UI_GRID_LINE': (150, 150, 150),  # Darker gray for better grid visibility
    'UI_TEXT': (50, 50, 50),          # Dark gray
    'UI_BUTTON': (100, 149, 237),     # Cornflower blue
    'UI_BUTTON_HOVER': (70, 130, 180), # Steel blue
    'UI_BUTTON_ACTIVE': (65, 105, 225), # Royal blue
    'UI_PAUSE': (144, 238, 144),       # Light green
    'UI_STOP': (255, 182, 193),        # Light pink
    'UI_CHART_BG': (255, 255, 255),    # White
    'UI_CHART_GRID': (230, 230, 230),  # Light gray
}

@dataclass
class ChartData:
    """Data structure for chart rendering"""
    values: List[float]
    max_value: float
    min_value: float
    color: Tuple[int, int, int]
    title: str

class PygameMonitor:
    def __init__(self, sim, cfg):
        self.sim = sim
        self.cfg = cfg
        
        # Initialize pygame
        pygame.init()
        
        # Display settings - much larger for better grid visibility
        self.width = 1920
        self.height = 1080
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Civilization Simulation - Pygame Monitor")
        
        # Fonts
        self.fonts = {
            'small': pygame.font.Font(None, 18),
            'medium': pygame.font.Font(None, 22),
            'large': pygame.font.Font(None, 28),
            'title': pygame.font.Font(None, 32)
        }
        
        # Layout constants - grid is now the main focus
        self.grid_size = 1200  # Much larger grid
        self.grid_x = 50
        self.grid_y = 50
        
        # Calculate cell size to fit the grid
        self.cell_size = min(self.grid_size // self.cfg.W, self.grid_size // self.cfg.H)
        
        # UI state
        self.is_paused = False
        self.should_stop = False
        self.mouse_pos = (0, 0)
        
        # Chart data
        self.charts = self._init_charts()
        
        # Button definitions
        self.buttons = self._init_buttons()
        
        # Legend
        self.legend = self._init_legend()
        
        # Performance tracking
        self.fps_clock = pygame.time.Clock()
        self.fps = 60
        
    def _init_charts(self):
        """Initialize chart data structures"""
        return {
            'population': ChartData([], 0, 0, (31, 119, 180), "Population"),
            'energy': ChartData([], 0, 0, (44, 160, 44), "Mean Energy"),
            'strength_dist': ChartData([], 0, 0, (255, 127, 0), "Strength Distribution"),
            'daily_stats': ChartData([], 0, 0, (128, 0, 128), "Daily Statistics")
        }
    
    def _init_buttons(self):
        """Initialize control buttons"""
        button_width = 120
        button_height = 40
        button_y = self.height - 60
        
        return {
            'pause_play': {
                'rect': pygame.Rect(self.width//2 - button_width - 20, button_y, button_width, button_height),
                'text': '⏸ Pause',
                'color': COLORS['UI_BUTTON'],
                'hover_color': COLORS['UI_BUTTON_HOVER'],
                'action': 'toggle_pause'
            },
            'stop': {
                'rect': pygame.Rect(self.width//2 + 20, button_y, button_width, button_height),
                'text': '⏹ Stop',
                'color': COLORS['UI_STOP'],
                'hover_color': (255, 150, 150),
                'action': 'stop'
            }
        }
    
    def _init_legend(self):
        """Initialize legend items - positioned to the right of the grid"""
        legend_x = self.grid_x + self.grid_size + 20
        legend_y = 50
        legend_width = 250
        
        items = [
            # Food zones
            {'type': 'zone', 'color': COLORS['ZONE_LOW'], 'label': 'Low Food Zone'},
            {'type': 'zone', 'color': COLORS['ZONE_MID'], 'label': 'Medium Food Zone'},
            {'type': 'zone', 'color': COLORS['ZONE_HIGH'], 'label': 'High Food Zone'},
            
            # Carrots
            {'type': 'carrot', 'color': COLORS['CARROT'], 'label': 'Carrots'},
            
            # Agent energy levels
            {'type': 'energy', 'color': COLORS['ENERGY_LOW'], 'label': 'Low Energy'},
            {'type': 'energy', 'color': COLORS['ENERGY_MID'], 'label': 'Medium Energy'},
            {'type': 'energy', 'color': COLORS['ENERGY_HIGH'], 'label': 'High Energy'},
            
            # Agent actions
            {'type': 'action', 'color': COLORS['ACTION_SEEK'], 'label': 'Seeking Food'},
            {'type': 'action', 'color': COLORS['ACTION_EXPLORE'], 'label': 'Exploring'},
            {'type': 'action', 'color': COLORS['ACTION_FORAGE'], 'label': 'Foraging'},
            {'type': 'action', 'color': COLORS['ACTION_COMBAT'], 'label': 'In Combat'}
        ]
        
        return {
            'x': legend_x,
            'y': legend_y,
            'width': legend_width,
            'items': items
        }
    
    def _draw_grid(self):
        """Draw the simulation grid with zones, carrots, and agents"""
        # Draw food zones
        for y in range(self.cfg.H):
            for x in range(self.cfg.W):
                zone_level = self.sim.world.richness[y, x]
                if zone_level == 0:
                    color = COLORS['ZONE_LOW']
                elif zone_level == 1:
                    color = COLORS['ZONE_MID']
                else:
                    color = COLORS['ZONE_HIGH']
                
                rect = pygame.Rect(
                    self.grid_x + x * self.cell_size,
                    self.grid_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)
                # Darker grid lines for better visibility
                pygame.draw.rect(self.screen, COLORS['UI_GRID_LINE'], rect, 1)
        
        # Draw carrots
        carrot_positions = np.nonzero(self.sim.world.carrots)
        for y, x in zip(carrot_positions[0], carrot_positions[1]):
            center_x = self.grid_x + x * self.cell_size + self.cell_size // 2
            center_y = self.grid_y + y * self.cell_size + self.cell_size // 2
            size = max(3, self.cell_size // 3)
            
            # Draw triangle (carrot)
            points = [
                (center_x, center_y - size),
                (center_x - size, center_y + size),
                (center_x + size, center_y + size)
            ]
            pygame.draw.polygon(self.screen, COLORS['CARROT'], points)
        
        # Draw agents
        if self.sim.agents:
            for agent in self.sim.agents:
                # Agent position
                agent_x = self.grid_x + agent.x * self.cell_size + self.cell_size // 2
                agent_y = self.grid_y + agent.y * self.cell_size + self.cell_size // 2
                agent_radius = max(2, self.cell_size // 3)
                
                # Energy-based fill color
                energy_frac = agent.energy / max(agent.genome.max_energy, 1e-6)
                energy_frac = np.clip(energy_frac, 0, 1)
                
                if energy_frac < 0.33:
                    outline_color = COLORS['ENERGY_LOW']
                elif energy_frac < 0.66:
                    outline_color = COLORS['ENERGY_MID']
                else:
                    outline_color = COLORS['ENERGY_HIGH']
                
                # Action-based outline color
                if agent.action == "SEEK":
                    fill_color = COLORS['ACTION_SEEK']
                elif agent.action == "FORAGE":
                    fill_color = COLORS['ACTION_FORAGE']
                elif agent.action == "COMBAT":
                    fill_color = COLORS['ACTION_COMBAT']
                else:
                    fill_color = COLORS['ACTION_EXPLORE']
                
                # Draw agent
                pygame.draw.circle(self.screen, fill_color, (agent_x, agent_y), agent_radius)
                pygame.draw.circle(self.screen, outline_color, (agent_x, agent_y), agent_radius, 2)
    
    def _draw_charts(self):
        """Draw real-time charts - positioned below the legend"""
        chart_x = self.legend['x']
        chart_y = self.legend['y'] + 320  # Reduced spacing
        chart_width = 250
        chart_height = 120  # Smaller charts
        
        # Population chart
        self._draw_line_chart(
            chart_x, chart_y, chart_width, chart_height,
            self.charts['population'], "Population"
        )
        
        # Energy chart
        self._draw_line_chart(
            chart_x, chart_y + chart_height + 15, chart_width, chart_height,
            self.charts['energy'], "Mean Energy"
        )
        
        # Strength distribution chart
        self._draw_histogram_chart(
            chart_x, chart_y + 2 * (chart_height + 15), chart_width, chart_height,
            self.charts['strength_dist'], "Strength Distribution"
        )
        
        # Daily statistics chart
        self._draw_daily_stats_chart(
            chart_x, chart_y + 3 * (chart_height + 15), chart_width, chart_height,
            self.charts['daily_stats'], "Daily Statistics"
        )
    
    def _draw_line_chart(self, x, y, width, height, data, title):
        """Draw a line chart"""
        # Background
        chart_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, COLORS['UI_CHART_BG'], chart_rect)
        pygame.draw.rect(self.screen, COLORS['UI_BORDER'], chart_rect, 2)
        
        # Title
        title_surface = self.fonts['small'].render(title, True, COLORS['UI_TEXT'])
        self.screen.blit(title_surface, (x + 5, y + 5))
        
        if len(data.values) < 2:
            return
        
        # Grid lines
        for i in range(5):
            grid_y = y + 20 + (i * (height - 40) // 4)
            pygame.draw.line(self.screen, COLORS['UI_CHART_GRID'], 
                           (x, grid_y), (x + width, grid_y), 1)
        
        # Data line
        if len(data.values) > 1:
            points = []
            for i, value in enumerate(data.values):
                if data.max_value > data.min_value:
                    norm_value = (value - data.min_value) / (data.max_value - data.min_value)
                else:
                    norm_value = 0.5
                
                point_x = x + 10 + (i * (width - 20) / max(1, len(data.values) - 1))
                point_y = y + height - 20 - (norm_value * (height - 40))
                points.append((point_x, point_y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, data.color, False, points, 2)
    
    def _draw_histogram_chart(self, x, y, width, height, data, title):
        """Draw a histogram chart for strength distribution"""
        # Background
        chart_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, COLORS['UI_CHART_BG'], chart_rect)
        pygame.draw.rect(self.screen, COLORS['UI_BORDER'], chart_rect, 2)
        
        # Title
        title_surface = self.fonts['small'].render(title, True, COLORS['UI_TEXT'])
        self.screen.blit(title_surface, (x + 5, y + 5))
        
        if not self.sim.agents:
            return
        
        # Get current strength values
        strengths = [agent.genome.strength for agent in self.sim.agents if agent.alive]
        if not strengths:
            return
        
        # Create histogram
        min_strength = min(strengths)
        max_strength = max(strengths)
        if max_strength == min_strength:
            return
        
        # Bin the data
        num_bins = 10
        bin_width = (width - 20) // num_bins
        bins = [0] * num_bins
        
        for strength in strengths:
            bin_idx = min(int((strength - min_strength) / (max_strength - min_strength) * num_bins), num_bins - 1)
            bins[bin_idx] += 1
        
        max_count = max(bins) if bins else 1
        
        # Draw histogram bars
        for i, count in enumerate(bins):
            if count > 0:
                bar_height = (count / max_count) * (height - 40)
                bar_x = x + 10 + i * bin_width
                bar_y = y + height - 20 - bar_height
                
                bar_rect = pygame.Rect(bar_x, bar_y, bin_width - 1, bar_height)
                pygame.draw.rect(self.screen, data.color, bar_rect)
                pygame.draw.rect(self.screen, COLORS['UI_BORDER'], bar_rect, 1)
    
    def _draw_daily_stats_chart(self, x, y, width, height, data, title):
        """Draw daily statistics chart"""
        # Background
        chart_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, COLORS['UI_CHART_BG'], chart_rect)
        pygame.draw.rect(self.screen, COLORS['UI_BORDER'], chart_rect, 2)
        
        # Title
        title_surface = self.fonts['small'].render(title, True, COLORS['UI_TEXT'])
        self.screen.blit(title_surface, (x + 5, y + 5))
        
        # Get today's stats
        fights_today = getattr(self.sim, 'fights_today', 0)
        deaths_today = getattr(self.sim, 'deaths_today', 0)
        births_today = getattr(self.sim, 'births_today', 0)
        carrots_today = int(self.sim.world.carrots.sum())
        
        # Display stats
        y_offset = 25
        stats = [
            f"Fights: {fights_today}",
            f"Deaths: {deaths_today}",
            f"Births: {births_today}",
            f"Carrots: {carrots_today}"
        ]
        
        for stat in stats:
            stat_surface = self.fonts['small'].render(stat, True, COLORS['UI_TEXT'])
            self.screen.blit(stat_surface, (x + 10, y + y_offset))
            y_offset += 20
    
    def _draw_legend(self):
        """Draw the map legend"""
        # Legend background
        legend_rect = pygame.Rect(self.legend['x'], self.legend['y'], 
                                self.legend['width'], 300)
        pygame.draw.rect(self.screen, COLORS['UI_CHART_BG'], legend_rect)
        pygame.draw.rect(self.screen, COLORS['UI_BORDER'], legend_rect, 2)
        
        # Legend title
        title_surface = self.fonts['medium'].render("Map Legend", True, COLORS['UI_TEXT'])
        self.screen.blit(title_surface, (self.legend['x'] + 10, self.legend['y'] + 10))
        
        # Legend items
        y_offset = 50
        for item in self.legend['items']:
            item_y = self.legend['y'] + y_offset
            
            if item['type'] == 'zone':
                # Zone color patch
                patch_rect = pygame.Rect(self.legend['x'] + 10, item_y, 20, 20)
                pygame.draw.rect(self.screen, item['color'], patch_rect)
                pygame.draw.rect(self.screen, COLORS['UI_BORDER'], patch_rect, 1)
                
            elif item['type'] == 'carrot':
                # Carrot triangle
                center_x = self.legend['x'] + 20
                center_y = item_y + 10
                size = 8
                points = [
                    (center_x, center_y - size),
                    (center_x - size, center_y + size),
                    (center_x + size, center_y + size)
                ]
                pygame.draw.polygon(self.screen, item['color'], points)
                
            elif item['type'] == 'action':
                # Action circle
                center_x = self.legend['x'] + 20
                center_y = item_y + 10
                pygame.draw.circle(self.screen, item['color'], (center_x, center_y), 8)
                
            elif item['type'] == 'energy':
                # Energy outline circle
                center_x = self.legend['x'] + 20
                center_y = item_y + 10
                pygame.draw.circle(self.screen, item['color'], (center_x, center_y), 8, 2)
            
            # Label
            label_surface = self.fonts['small'].render(item['label'], True, COLORS['UI_TEXT'])
            self.screen.blit(label_surface, (self.legend['x'] + 40, item_y))
            
            y_offset += 25
    
    def _draw_buttons(self):
        """Draw control buttons"""
        for button_id, button in self.buttons.items():
            # Check if mouse is hovering
            if button['rect'].collidepoint(self.mouse_pos):
                color = button['hover_color']
            else:
                color = button['color']
            
            # Draw button
            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, COLORS['UI_BORDER'], button['rect'], 2)
            
            # Button text
            text_surface = self.fonts['medium'].render(button['text'], True, COLORS['UI_TEXT'])
            text_rect = text_surface.get_rect(center=button['rect'].center)
            self.screen.blit(text_surface, text_rect)
    
    def _draw_status(self):
        """Draw simulation status information"""
        status_x = 50
        status_y = self.grid_y + self.grid_size + 20
        
        # Status background
        status_rect = pygame.Rect(status_x, status_y, self.grid_size, 60)
        pygame.draw.rect(self.screen, COLORS['UI_CHART_BG'], status_rect)
        pygame.draw.rect(self.screen, COLORS['UI_BORDER'], status_rect, 2)
        
        # Status text
        if self.should_stop:
            status_text = f"Simulation Stopped - Day {len(self.sim.day_population)}, Tick {self.sim.fights_today}"
            status_color = COLORS['UI_STOP']
        elif self.is_paused:
            status_text = f"Paused - Day {len(self.sim.day_population)}, Tick {self.sim.fights_today}"
            status_color = COLORS['UI_PAUSE']
        else:
            status_text = f"Running - Day {len(self.sim.day_population)}, Tick {self.sim.fights_today}"
            status_color = COLORS['UI_BUTTON']
        
        status_surface = self.fonts['medium'].render(status_text, True, status_color)
        self.screen.blit(status_surface, (status_x + 10, status_y + 10))
        
        # Population info
        pop_text = f"Population: {len(self.sim.agents)} | Carrots: {int(self.sim.world.carrots.sum())}"
        pop_surface = self.fonts['small'].render(pop_text, True, COLORS['UI_TEXT'])
        self.screen.blit(pop_surface, (status_x + 10, status_y + 35))
    
    def _update_chart_data(self):
        """Update chart data from simulation"""
        # Population
        if self.sim.day_population:
            self.charts['population'].values = self.sim.day_population
            self.charts['population'].max_value = max(self.sim.day_population)
            self.charts['population'].min_value = min(self.sim.day_population)
        
        # Energy
        if self.sim.day_mean_energy:
            self.charts['energy'].values = self.sim.day_mean_energy
            self.charts['energy'].max_value = max(self.sim.day_mean_energy)
            self.charts['energy'].min_value = min(self.sim.day_mean_energy)
        
        # Strength distribution (current live data)
        if self.sim.agents:
            strengths = [agent.genome.strength for agent in self.sim.agents if agent.alive]
            if strengths:
                self.charts['strength_dist'].values = strengths
                self.charts['strength_dist'].max_value = max(strengths)
                self.charts['strength_dist'].min_value = min(strengths)
        
        # Daily statistics (current day's data)
        fights_today = getattr(self.sim, 'fights_today', 0)
        deaths_today = getattr(self.sim, 'deaths_today', 0)
        births_today = getattr(self.sim, 'births_today', 0)
        
        # Store daily stats for the chart
        if not self.charts['daily_stats'].values:
            self.charts['daily_stats'].values = []
        
        # Update with current day's stats
        daily_total = fights_today + deaths_today + births_today
        self.charts['daily_stats'].values.append(daily_total)
        if len(self.charts['daily_stats'].values) > 20:
            self.charts['daily_stats'].values.pop(0)
        self.charts['daily_stats'].max_value = max(self.charts['daily_stats'].values) if self.charts['daily_stats'].values else 1
    
    def _handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_stop = True
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_click(event.pos)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._toggle_pause()
                elif event.key == pygame.K_s:
                    self._stop_simulation()
                elif event.key == pygame.K_ESCAPE:
                    self.should_stop = True
                    return False
            
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
        
        return True
    
    def _handle_click(self, pos):
        """Handle mouse clicks on buttons"""
        for button_id, button in self.buttons.items():
            if button['rect'].collidepoint(pos):
                if button['action'] == 'toggle_pause':
                    self._toggle_pause()
                elif button['action'] == 'stop':
                    self._stop_simulation()
                break
    
    def _toggle_pause(self):
        """Toggle pause/play state"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.buttons['pause_play']['text'] = '▶ Play'
            self.buttons['pause_play']['color'] = COLORS['UI_PAUSE']
        else:
            self.buttons['pause_play']['text'] = '⏸ Pause'
            self.buttons['pause_play']['color'] = COLORS['UI_BUTTON']
    
    def _stop_simulation(self):
        """Stop the simulation"""
        self.should_stop = True
        self.buttons['stop']['text'] = '⏹ Stopped'
        self.buttons['stop']['color'] = (255, 100, 100)
    
    def render(self, day_idx, tick_idx):
        """Main render method called each tick"""
        # Handle events
        if not self._handle_events():
            return False
        
        # Update chart data
        self._update_chart_data()
        
        # Clear screen
        self.screen.fill(COLORS['UI_BACKGROUND'])
        
        # Draw all components
        self._draw_grid()
        self._draw_legend()
        self._draw_charts()
        self._draw_buttons()
        self._draw_status()
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        self.fps_clock.tick(self.fps)
        
        return True
    
    def should_continue(self):
        """Check if simulation should continue"""
        return not self.should_stop
    
    def cleanup(self):
        """Clean up pygame resources"""
        pygame.quit()
