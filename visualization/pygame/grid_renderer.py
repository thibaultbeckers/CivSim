import pygame
import numpy as np
from .colors import COLORS

def draw_grid(monitor):
    """Draw the simulation grid with zones, carrots, and agents"""
    # Draw food zones
    for y in range(monitor.cfg.H):
        for x in range(monitor.cfg.W):
            zone_level = monitor.sim.world.richness[y, x]
            if zone_level == 0:
                color = COLORS['ZONE_LOW']
            elif zone_level == 1:
                color = COLORS['ZONE_MID']
            else:
                color = COLORS['ZONE_HIGH']
            
            rect = pygame.Rect(
                monitor.grid_x + x * monitor.cell_size,
                monitor.grid_y + y * monitor.cell_size,
                monitor.cell_size,
                monitor.cell_size
            )
            pygame.draw.rect(monitor.screen, color, rect)
            # Darker grid lines for better visibility
            pygame.draw.rect(monitor.screen, COLORS['UI_GRID_LINE'], rect, 1)
    
    # Draw carrots
    carrot_positions = np.nonzero(monitor.sim.world.carrots)
    for y, x in zip(carrot_positions[0], carrot_positions[1]):
        center_x = monitor.grid_x + x * monitor.cell_size + monitor.cell_size // 2
        center_y = monitor.grid_y + y * monitor.cell_size + monitor.cell_size // 2
        size = max(3, monitor.cell_size // 3)
        
        # Draw triangle (carrot)
        points = [
            (center_x, center_y - size),
            (center_x - size, center_y + size),
            (center_x + size, center_y + size)
        ]
        pygame.draw.polygon(monitor.screen, COLORS['CARROT'], points)
    
    # Draw agents
    if monitor.sim.agents:
        for agent in monitor.sim.agents:
            # Agent position
            agent_x = monitor.grid_x + agent.x * monitor.cell_size + monitor.cell_size // 2
            agent_y = monitor.grid_y + agent.y * monitor.cell_size + monitor.cell_size // 2
            agent_radius = max(2, monitor.cell_size // 3)
            
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
            pygame.draw.circle(monitor.screen, fill_color, (agent_x, agent_y), agent_radius)
            pygame.draw.circle(monitor.screen, outline_color, (agent_x, agent_y), agent_radius, 2)
