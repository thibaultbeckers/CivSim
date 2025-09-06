import pygame
from .colors import COLORS

def draw_legend(monitor):
    rect = pygame.Rect(monitor.legend['x'], monitor.legend['y'], monitor.legend['width'], 300)
    pygame.draw.rect(monitor.screen, COLORS['UI_CHART_BG'], rect)
    pygame.draw.rect(monitor.screen, COLORS['UI_BORDER'], rect, 2)

    title_surface = monitor.fonts['medium'].render("Map Legend", True, COLORS['UI_TEXT'])
    monitor.screen.blit(title_surface, (monitor.legend['x'] + 10, monitor.legend['y'] + 10))

    y_offset = 50
    for item in monitor.legend['items']:
        item_y = monitor.legend['y'] + y_offset
        if item['type'] == 'zone':
            patch = pygame.Rect(monitor.legend['x'] + 10, item_y, 20, 20)
            pygame.draw.rect(monitor.screen, item['color'], patch)
            pygame.draw.rect(monitor.screen, COLORS['UI_BORDER'], patch, 1)
        elif item['type'] == 'carrot':
            cx, cy, size = monitor.legend['x'] + 20, item_y + 10, 8
            points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
            pygame.draw.polygon(monitor.screen, item['color'], points)
        elif item['type'] == 'action':
            pygame.draw.circle(monitor.screen, item['color'], (monitor.legend['x'] + 20, item_y + 10), 8)
        elif item['type'] == 'energy':
            pygame.draw.circle(monitor.screen, item['color'], (monitor.legend['x'] + 20, item_y + 10), 8, 2)

        label = monitor.fonts['small'].render(item['label'], True, COLORS['UI_TEXT'])
        monitor.screen.blit(label, (monitor.legend['x'] + 40, item_y))
        y_offset += 25


def draw_buttons(monitor):
    for button in monitor.buttons.values():
        color = button['hover_color'] if button['rect'].collidepoint(monitor.mouse_pos) else button['color']
        pygame.draw.rect(monitor.screen, color, button['rect'])
        pygame.draw.rect(monitor.screen, COLORS['UI_BORDER'], button['rect'], 2)
        text_surface = monitor.fonts['medium'].render(button['text'], True, COLORS['UI_TEXT'])
        text_rect = text_surface.get_rect(center=button['rect'].center)
        monitor.screen.blit(text_surface, text_rect)


def draw_status(monitor):
    status_x = 50
    status_y = monitor.grid_y + monitor.grid_size + 20
    status_rect = pygame.Rect(status_x, status_y, monitor.grid_size, 100)
    pygame.draw.rect(monitor.screen, COLORS['UI_CHART_BG'], status_rect)
    pygame.draw.rect(monitor.screen, COLORS['UI_BORDER'], status_rect, 2)

    if monitor.should_stop:
        status_text = f"Simulation Stopped - Day {len(monitor.sim.day_population)}"
        status_color = COLORS['UI_STOP']
    elif monitor.is_paused:
        status_text = f"Paused - Day {len(monitor.sim.day_population)}"
        status_color = COLORS['UI_PAUSE']
    else:
        status_text = f"Running - Day {len(monitor.sim.day_population)}"
        status_color = COLORS['UI_BUTTON']

    status_surface = monitor.fonts['medium'].render(status_text, True, status_color)
    monitor.screen.blit(status_surface, (status_x + 10, status_y + 10))

    y_offset = 35
    pop_text = f"Population: {len(monitor.sim.agents)} | Carrots: {int(monitor.sim.world.carrots.sum())}"
    monitor.screen.blit(monitor.fonts['small'].render(pop_text, True, COLORS['UI_TEXT']), (status_x + 10, status_y + y_offset))

    events_text = f"Today: Fights {monitor.sim.fights_today} | Deaths {monitor.sim.deaths_today} | Births {monitor.sim.births_today}"
    monitor.screen.blit(monitor.fonts['small'].render(events_text, True, COLORS['UI_TEXT']), (status_x + 10, status_y + y_offset + 20))

    speed_text = f"Speed: {monitor.fps} FPS"
    surf = monitor.fonts['small'].render(speed_text, True, COLORS['UI_TEXT'])
    monitor.screen.blit(surf, (status_x + 10, status_y + y_offset + 40))


def draw_live_counters(monitor):
    counter_rect = pygame.Rect(monitor.grid_x, 20, monitor.grid_size, 40)
    pygame.draw.rect(monitor.screen, COLORS['UI_CHART_BG'], counter_rect)
    pygame.draw.rect(monitor.screen, COLORS['UI_BORDER'], counter_rect, 2)

    day, tick, total = len(monitor.sim.day_population), monitor.sim.fights_today, len(monitor.sim.day_population) * monitor.cfg.T_DAY + monitor.sim.fights_today
    text = f"Day {day} | Tick {tick} | Total Ticks {total} | Population {len(monitor.sim.agents)} | Carrots {int(monitor.sim.world.carrots.sum())}"
    surf = monitor.fonts['large'].render(text, True, COLORS['UI_TEXT'])
    rect = surf.get_rect(center=counter_rect.center)
    monitor.screen.blit(surf, rect)


def draw_corner_overlay(monitor):
    x, y, w, h = monitor.grid_x + monitor.grid_size - 200, monitor.grid_y + 10, 190, 80
    overlay = pygame.Surface((w, h))
    overlay.set_alpha(200)
    overlay.fill(COLORS['UI_CHART_BG'])
    monitor.screen.blit(overlay, (x, y))
    pygame.draw.rect(monitor.screen, COLORS['UI_BORDER'], (x, y, w, h), 2)

    day, tick, pop = len(monitor.sim.day_population), monitor.sim.fights_today, len(monitor.sim.agents)
    monitor.screen.blit(monitor.fonts['medium'].render(f"Day: {day}", True, COLORS['UI_TEXT']), (x + 10, y + 10))
    monitor.screen.blit(monitor.fonts['medium'].render(f"Tick: {tick}", True, COLORS['UI_TEXT']), (x + 10, y + 30))
    monitor.screen.blit(monitor.fonts['medium'].render(f"Pop: {pop}", True, COLORS['UI_TEXT']), (x + 10, y + 50))
