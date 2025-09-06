import time
from simulation.config import CFG
from simulation.sim import Sim
from simulation.charts import final_charts
from visualization.pygame.monitor import PygameMonitor


def run():
    cfg = CFG()
    sim = Sim(cfg)
    monitor = PygameMonitor(sim, cfg)

    tcount = 0
    last_tick_time = time.time()

    try:
        for day in range(cfg.N_DAYS):
            t = 0
            while t < cfg.T_DAY:
                now = time.time()
                tick_interval = 1.0 / monitor.ticks_per_second if monitor.ticks_per_second > 0 else float("inf")

                # only advance sim if enough time has passed
                if now - last_tick_time >= tick_interval:
                    last_tick_time += tick_interval

                    # stop / pause checks
                    if monitor.should_stop:
                        print("Simulation stopped by user")
                        final_charts(sim)
                        return
                    while monitor.is_paused and not monitor.should_stop:
                        if not monitor.render(day, t):
                            return
                    if monitor.should_stop:
                        print("Simulation stopped by user")
                        final_charts(sim)
                        return

                    # simulation tick
                    sim.upkeep()
                    if not sim.agents:
                        monitor.render(day, t)
                        break
                    sim.decide_and_move()
                    if not sim.agents:
                        monitor.render(day, t)
                        break
                    sim.forage()
                    sim.combat()
                    sim.world.spawn_carrots()

                    t += 1
                    tcount += 1

                # always render monitor
                if not monitor.render(day, t):
                    return

            sim.night_phase(day)
            sim.record_day_metrics()

        if not monitor.should_stop:
            print("Simulation completed normally")
        final_charts(sim)

    finally:
        monitor.cleanup()


if __name__ == "__main__":
    run()
