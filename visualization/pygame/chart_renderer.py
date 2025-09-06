import pygame
from .colors import COLORS

def update_chart_data(monitor):
    """Update chart data from simulation"""
    # Population
    if monitor.sim.day_population:
        monitor.charts['population'].values = monitor.sim.day_population
        monitor.charts['population'].max_value = max(monitor.sim.day_population)
        monitor.charts['population'].min_value = min(monitor.sim.day_population)

    # Energy
    if monitor.sim.day_mean_energy:
        monitor.charts['energy'].values = monitor.sim.day_mean_energy
        monitor.charts['energy'].max_value = max(monitor.sim.day_mean_energy)
        monitor.charts['energy'].min_value = min(monitor.sim.day_mean_energy)

    # Perception distribution
    if monitor.sim.agents:
        perceptions = [a.genome.perception for a in monitor.sim.agents if a.alive]
        if perceptions:
            monitor.charts['perception_dist'].values = perceptions
            monitor.charts['perception_dist'].max_value = max(perceptions)
            monitor.charts['perception_dist'].min_value = min(perceptions)

    # Strength distribution
    if monitor.sim.agents:
        strengths = [a.genome.strength for a in monitor.sim.agents if a.alive]
        if strengths:
            monitor.charts['strength_dist'].values = strengths
            monitor.charts['strength_dist'].max_value = max(strengths)
            monitor.charts['strength_dist'].min_value = min(strengths)

    # Aggression distribution
    if monitor.sim.agents:
        aggressions = [a.genome.aggression for a in monitor.sim.agents if a.alive]
        if aggressions:
            monitor.charts['aggression_dist'].values = aggressions
            monitor.charts['aggression_dist'].max_value = max(aggressions)
            monitor.charts['aggression_dist'].min_value = min(aggressions)

    # Daily statistics
    fights_today = getattr(monitor.sim, 'fights_today', 0)
    deaths_today = getattr(monitor.sim, 'deaths_today', 0)
    births_today = getattr(monitor.sim, 'births_today', 0)

    if not monitor.charts['daily_stats'].values:
        monitor.charts['daily_stats'].values = []

    daily_total = fights_today + deaths_today + births_today
    monitor.charts['daily_stats'].values.append(daily_total)
    if len(monitor.charts['daily_stats'].values) > 20:
        monitor.charts['daily_stats'].values.pop(0)
    monitor.charts['daily_stats'].max_value = max(monitor.charts['daily_stats'].values) if monitor.charts['daily_stats'].values else 1


# ========== Drawing helpers ==========

def _draw_line_chart(monitor, x, y, width, height, data, title):
    rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(monitor.screen, COLORS['UI_CHART_BG'], rect)
    pygame.draw.rect(monitor.screen, COLORS['UI_BORDER'], rect, 2)

    title_surface = monitor.fonts['small'].render(title, True, COLORS['UI_TEXT'])
    monitor.screen.blit(title_surface, (x + 5, y + 5))

    if len(data.values) < 2:
        return

    # Grid lines
    for i in range(5):
        grid_y = y + 20 + (i * (height - 40) // 4)
        pygame.draw.line(monitor.screen, COLORS['UI_CHART_GRID'], (x, grid_y), (x + width, grid_y), 1)

    # Data line
    points = []
    for i, value in enumerate(data.values):
        norm = (value - data.min_value) / (data.max_value - data.min_value) if data.max_value > data.min_value else 0.5
        px = x + 10 + (i * (width - 20) / max(1, len(data.values) - 1))
        py = y + height - 20 - (norm * (height - 40))
        points.append((px, py))
    if len(points) > 1:
        pygame.draw.lines(monitor.screen, data.color, False, points, 2)


def _draw_histogram_chart(monitor, x, y, width, height, data, title):
    rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(monitor.screen, COLORS['UI_CHART_BG'], rect)
    pygame.draw.rect(monitor.screen, COLORS['UI_BORDER'], rect, 2)

    title_surface = monitor.fonts['small'].render(title, True, COLORS['UI_TEXT'])
    monitor.screen.blit(title_surface, (x + 5, y + 5))

    if not data.values:
        return

    values = data.values
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return

    num_bins = 10
    bin_width = (width - 20) // num_bins
    bins = [0] * num_bins
    for v in values:
        idx = min(int((v - min_val) / (max_val - min_val) * num_bins), num_bins - 1)
        bins[idx] += 1

    max_count = max(bins) or 1
    for i, count in enumerate(bins):
        if count > 0:
            bar_height = (count / max_count) * (height - 40)
            bar_rect = pygame.Rect(x + 10 + i * bin_width, y + height - 20 - bar_height, bin_width - 1, bar_height)
            pygame.draw.rect(monitor.screen, data.color, bar_rect)
            pygame.draw.rect(monitor.screen, COLORS['UI_BORDER'], bar_rect, 1)


def _draw_daily_stats_chart(monitor, x, y, width, height, title):
    rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(monitor.screen, COLORS['UI_CHART_BG'], rect)
    pygame.draw.rect(monitor.screen, COLORS['UI_BORDER'], rect, 2)

    title_surface = monitor.fonts['small'].render(title, True, COLORS['UI_TEXT'])
    monitor.screen.blit(title_surface, (x + 5, y + 5))

    stats = [
        f"Fights: {getattr(monitor.sim, 'fights_today', 0)}",
        f"Deaths: {getattr(monitor.sim, 'deaths_today', 0)}",
        f"Births: {getattr(monitor.sim, 'births_today', 0)}",
        f"Carrots: {int(monitor.sim.world.carrots.sum())}"
    ]
    y_offset = 25
    for stat in stats:
        surf = monitor.fonts['small'].render(stat, True, COLORS['UI_TEXT'])
        monitor.screen.blit(surf, (x + 10, y + y_offset))
        y_offset += 20


# ========== Main drawing orchestrator ==========

def draw_charts(monitor):
    chart_x = monitor.legend['x']
    chart_y = monitor.legend['y'] + 320
    w, h = 250, 120

    _draw_line_chart(monitor, chart_x, chart_y, w, h, monitor.charts['population'], "Population")
    _draw_line_chart(monitor, chart_x, chart_y + h + 15, w, h, monitor.charts['energy'], "Mean Energy")
    _draw_histogram_chart(monitor, chart_x, chart_y + 2 * (h + 15), w, h, monitor.charts['strength_dist'], "Strength Dist.")
    _draw_histogram_chart(monitor, chart_x, chart_y + 3 * (h + 15), w, h, monitor.charts['perception_dist'], "Perception Dist.")
    _draw_histogram_chart(monitor, chart_x, chart_y + 4 * (h + 15), w, h, monitor.charts['aggression_dist'], "Aggression Dist.")
    _draw_daily_stats_chart(monitor, chart_x, chart_y + 5 * (h + 15), w, h, "Daily Stats")
