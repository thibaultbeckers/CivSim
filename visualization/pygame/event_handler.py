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
            action = button['action']
            if action == "toggle_pause":
                _toggle_pause(monitor)
            elif action == "stop":
                _stop_simulation(monitor)
            elif action == "speed_up":
                monitor.ticks_per_second = min(monitor.ticks_per_second * 2, 60.0)
                print(f"Simulation speed: {monitor.ticks_per_second:.1f} ticks/sec")
            elif action == "slow_down":
                monitor.ticks_per_second = max(monitor.ticks_per_second / 2, 0.125)
                print(f"Simulation speed: {monitor.ticks_per_second:.3f} ticks/sec")


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
