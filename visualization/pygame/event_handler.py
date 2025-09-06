# event_handler.py
import pygame
from .colors import COLORS

def handle_events(monitor):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            monitor.should_stop = True
            return False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            _handle_click(monitor, event.pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                _toggle_pause(monitor)
            elif event.key == pygame.K_s:
                _stop_simulation(monitor)
            elif event.key == pygame.K_ESCAPE:
                monitor.should_stop = True
                return False
        elif event.type == pygame.MOUSEMOTION:
            monitor.mouse_pos = event.pos
    return True

def _handle_click(monitor, pos):
    for button in monitor.buttons.values():
        if button['rect'].collidepoint(pos):
            if button['action'] == 'toggle_pause':
                _toggle_pause(monitor)
            elif button['action'] == 'stop':
                _stop_simulation(monitor)
            break

def _toggle_pause(monitor):
    monitor.is_paused = not monitor.is_paused
    if monitor.is_paused:
        monitor.buttons['pause_play']['text'] = '▶ Play'
        monitor.buttons['pause_play']['color'] = COLORS['UI_PAUSE']
    else:
        monitor.buttons['pause_play']['text'] = '⏸ Pause'
        monitor.buttons['pause_play']['color'] = COLORS['UI_BUTTON']

def _stop_simulation(monitor):
    monitor.should_stop = True
    monitor.buttons['stop']['text'] = '⏹ Stopped'
    monitor.buttons['stop']['color'] = (255, 100, 100)
