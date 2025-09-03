# main.py
# Entry point for running the simulation with pygame visualization

from simulation.config import CFG
from simulation.sim import Sim
from simulation.charts import final_charts
from visualization.pygame_monitor import PygameMonitor


def run():
    cfg = CFG()
    sim = Sim(cfg)
    ui = PygameMonitor(sim, cfg)

    ticks_total = cfg.N_DAYS * cfg.T_DAY
    tcount = 0

    try:
        for day in range(cfg.N_DAYS):
            for t in range(cfg.T_DAY):
                # Check if simulation should stop
                if ui.should_stop:
                    print("Simulation stopped by user")
                    final_charts(sim)
                    return

                # Check if simulation is paused
                while ui.is_paused and not ui.should_stop:
                    if not ui.render(day, t):
                        return

                # Check again after unpausing
                if ui.should_stop:
                    print("Simulation stopped by user")
                    final_charts(sim)
                    return

                sim.upkeep()
                if not sim.agents:
                    ui.render(day, t)
                    break  # extinction
                sim.decide_and_move()
                if not sim.agents:
                    ui.render(day, t)
                    break
                sim.forage()
                sim.combat()
                sim.world.spawn_carrots()

                if (t % cfg.RENDER_EVERY) == 0:
                    if not ui.render(day, t):
                        return
                tcount += 1

            # Night phase (instant)
            sim.night_phase(day)
            sim.record_day_metrics()

        # End-of-sim charts
        if not ui.should_stop:
            print("Simulation completed normally")
        final_charts(sim)

    finally:
        # Always cleanup pygame
        ui.cleanup()


if __name__ == "__main__":
    run()
