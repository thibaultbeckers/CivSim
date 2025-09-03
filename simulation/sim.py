import math
import numpy as np
from typing import List, Tuple, Optional, Dict

from .config import CFG
from .entities import Agent, Genome
from .utils import clamp, wrap, torus_manhattan
from .world import World


class Sim:
    def __init__(self, cfg: CFG):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.SEED)
        self.world = World(cfg, self.rng)
        self.agents: List[Agent] = []
        self.next_id = 0

        # Metrics
        self.day_population: List[int] = []
        self.day_mean_energy: List[float] = []
        self.day_fights: List[int] = []

        self.trait_series: Dict[str, List[np.ndarray]] = {k: [] for k in
            ["max_energy","metabolism","move_cost","perception","strength","aggression","repro_threshold","repro_cost"]}
        self.energy_series: List[float] = []

        # daily counters
        self.fights_today = 0
        self.births_today = 0
        self.deaths_today = 0

        self._init_agents()

    # ----- initialization -----

    def _init_agents(self):
        cfg = self.cfg
        for _ in range(cfg.N0):
            g = self._random_genome()
            e = self.rng.uniform(cfg.INIT_ENERGY_FRAC[0], cfg.INIT_ENERGY_FRAC[1]) * g.max_energy
            a = Agent(
                id=self._claim_id(), parent_id=None,
                x=int(self.rng.integers(0, cfg.W)),
                y=int(self.rng.integers(0, cfg.H)),
                energy=float(e), age=0, alive=True, genome=g
            )
            self.agents.append(a)
            print(f"[0] SPAWN id={a.id} parent=None pos=({a.x},{a.y}) genes={vars(a.genome)}")

    def _random_genome(self) -> Genome:
        cfg = self.cfg
        def sampr(lo, hi, default):
            return float(self.rng.uniform(lo, hi))
        def sampi(lo, hi, default):
            return int(self.rng.integers(lo, hi+1))
        return Genome(
            max_energy = sampr(*cfg.G_MAX_ENERGY),
            metabolism = sampr(*cfg.G_METABOLISM),
            move_cost  = sampr(*cfg.G_MOVE_COST),
            perception = sampi(*cfg.G_PERCEPTION),
            strength   = sampr(*cfg.G_STRENGTH),
            aggression = sampr(*cfg.G_AGGRESSION),
            repro_threshold = sampr(*cfg.G_REPRO_THRESH),
            repro_cost = sampr(*cfg.G_REPRO_COST)
        )

    def _mutate(self, g: Genome) -> Genome:
        cfg = self.cfg
        def mut(v, lo, hi):
            stdev = (hi - lo) * cfg.MUT_PCT
            return clamp(v + self.rng.normal(0, stdev), lo, hi)
        def muti(v, lo, hi):
            return int(round(mut(v, lo, hi)))
        return Genome(
            max_energy = mut(g.max_energy, *cfg.G_MAX_ENERGY[:2]),
            metabolism = mut(g.metabolism, *cfg.G_METABOLISM[:2]),
            move_cost  = mut(g.move_cost, *cfg.G_MOVE_COST[:2]),
            perception = max(cfg.G_PERCEPTION[0], min(cfg.G_PERCEPTION[1], int(round(g.perception + self.rng.normal(0, (cfg.G_PERCEPTION[1]-cfg.G_PERCEPTION[0])*cfg.MUT_PCT))))),
            strength   = mut(g.strength, *cfg.G_STRENGTH[:2]),
            aggression = mut(g.aggression, *cfg.G_AGGRESSION[:2]),
            repro_threshold = mut(g.repro_threshold, *cfg.G_REPRO_THRESH[:2]),
            repro_cost = mut(g.repro_cost, *cfg.G_REPRO_COST[:2]),
        )

    def _claim_id(self) -> int:
        nid = self.next_id
        self.next_id += 1
        return nid

    # ----- per-tick mechanics -----

    def _effective_metabolism(self, agent: Agent) -> float:
        if agent.age <= self.cfg.AGE_SOFT:
            return agent.genome.metabolism
        else:
            delta = agent.age - self.cfg.AGE_SOFT
            return agent.genome.metabolism * math.exp(self.cfg.AGE_EXP_K * delta)

    def upkeep(self):
        # age, metabolism drain, remove dead
        for a in self.agents:
            if not a.alive: continue
            a.age += 1
            a.energy -= self._effective_metabolism(a)
            if a.energy <= 0:
                a.alive = False
                self.deaths_today += 1
                print(f"[tick] DEATH id={a.id} cause=STARVE pos=({a.x},{a.y}) age={a.age}")
        # compact list
        self.agents = [a for a in self.agents if a.alive]

    def perceive_target_carrot(self, a: Agent) -> Optional[Tuple[int,int]]:
        """Find nearest carrot within Chebyshev radius perception; return (x,y) or None."""
        R = a.genome.perception
        H, W = self.cfg.H, self.cfg.W
        found = None
        bestd = 10**9
        for dy in range(-R, R+1):
            for dx in range(-R, R+1):
                x = wrap(a.x + dx, W)
                y = wrap(a.y + dy, H)
                if self.world.carrots[y, x]:
                    d = torus_manhattan(a.x, a.y, x, y, W, H)
                    if d < bestd:
                        bestd = d
                        found = (x, y)
        return found

    def step_toward(self, a: Agent, tx: int, ty: int) -> Tuple[int,int]:
        """One greedy von-Neumann step reducing Manhattan distance (ties random)."""
        W, H = self.cfg.W, self.cfg.H
        # candidate neighbors
        candidates = []
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = wrap(a.x+dx, W), wrap(a.y+dy, H)
            d = torus_manhattan(nx, ny, tx, ty, W, H)
            candidates.append((d, nx, ny))
        # choose min d, ties random
        mins = [c for c in candidates if c[0] == min(c[0] for c in candidates)]
        _, nx, ny = mins[self.rng.integers(0, len(mins))]
        return nx, ny

    def explore_step(self, a: Agent) -> Tuple[int,int, str]:
        """Bias toward higher richness in 4-neighborhood; if tie/equal -> pure random. Returns (nx,ny,reason)."""
        W, H = self.cfg.W, self.cfg.H
        rs = []
        maxR = -1
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = wrap(a.x+dx, W), wrap(a.y+dy, H)
            r = int(self.world.richness[ny, nx])
            rs.append((r, nx, ny))
            if r > maxR: maxR = r
        best = [t for t in rs if t[0] == maxR]
        if len(set(r for r,_,_ in rs)) == 1:
            # all equal -> pure random
            r, nx, ny = rs[self.rng.integers(0, len(rs))]
            return nx, ny, "RANDOM"
        else:
            r, nx, ny = best[self.rng.integers(0, len(best))]
            return nx, ny, "EXPLORE"

    def decide_and_move(self):
        cfg = self.cfg
        rng = self.rng
        rng.shuffle(self.agents)  # avoid bias each tick

        for a in self.agents:
            # Perceive
            target = self.perceive_target_carrot(a)
            if target is not None:
                nx, ny = self.step_toward(a, *target)
                # Log only SEEK moves
                print(f"[tick] MOVE id={a.id} from=({a.x},{a.y}) to=({nx},{ny}) reason=SEEK_CARROT")
                a.action = "SEEK"
            else:
                nx, ny, reason = self.explore_step(a)
                a.action = "EXPLORE" if reason == "EXPLORE" else "EXPLORE"  # RANDOM also maps to EXPLORE (not logged)
                # Do NOT log explore/random moves

            # Move + cost
            a.x, a.y = nx, ny
            a.energy -= a.genome.move_cost
            if a.energy <= 0:
                a.alive = False
                self.deaths_today += 1
                print(f"[tick] DEATH id={a.id} cause=STARVE pos=({a.x},{a.y}) age={a.age}")

        # compact list after movement
        self.agents = [a for a in self.agents if a.alive]

    def forage(self):
        for a in self.agents:
            if self.world.carrots[a.y, a.x]:
                a.energy = min(a.genome.max_energy, a.energy + self.cfg.E_CARROT)
                self.world.carrots[a.y, a.x] = False
                a.action = "FORAGE"
                print(f"[tick] FORAGE id={a.id} pos=({a.x},{a.y}) +E={self.cfg.E_CARROT:.1f} energy={a.energy:.1f}")

    def combat(self):
        # Build tile -> list of indices
        tilemap: Dict[Tuple[int,int], List[int]] = {}
        for i, a in enumerate(self.agents):
            tilemap.setdefault((a.x, a.y), []).append(i)

        to_kill: List[int] = []
        for (x, y), idxs in tilemap.items():
            if len(idxs) < 2:
                continue
            # probability of duel = mean aggression
            mean_aggr = float(np.mean([self.agents[i].genome.aggression for i in idxs]))
            if self.rng.random() > mean_aggr:
                continue

            # pick two random distinct agents on this tile
            if len(idxs) == 2:
                i1, i2 = idxs
            else:
                i1, i2 = self.rng.choice(idxs, size=2, replace=False)

            A = self.agents[i1]
            B = self.agents[i2]
            p1 = A.genome.strength * self.rng.uniform(0.9, 1.1)
            p2 = B.genome.strength * self.rng.uniform(0.9, 1.1)
            dmg = max(1, int(round(abs(p1 - p2))))
            if p1 >= p2:
                loser = B
                winner = A
            else:
                loser = A
                winner = B
            loser.energy -= dmg
            loser.action = "COMBAT"
            winner.action = "COMBAT"
            self.fights_today += 1
            print(f"[tick] COMBAT tile=({x},{y}) a={A.id} b={B.id} winner={winner.id} dmg={dmg}")
            if loser.energy <= 0:
                loser.alive = False
                self.deaths_today += 1
                print(f"[tick] DEATH id={loser.id} cause=COMBAT pos=({loser.x},{loser.y}) age={loser.age}")

        # compact after combat
        self.agents = [a for a in self.agents if a.alive]

    def night_phase(self, day_idx: int):
        newborns: List[Agent] = []
        for a in self.agents:
            g = a.genome
            if a.energy >= max(g.repro_threshold, g.repro_cost + 1.0):
                child_g = self._mutate(g)
                # place child in Moore neighborhood
                choices = [ (wrap(a.x+dx, self.cfg.W), wrap(a.y+dy, self.cfg.H))
                            for dx in (-1,0,1) for dy in (-1,0,1) if not (dx==0 and dy==0) ]
                cx, cy = choices[self.rng.integers(0, len(choices))]
                child_e = min(g.repro_cost, a.energy - g.repro_cost)
                a.energy -= g.repro_cost
                child = Agent(
                    id=self._claim_id(),
                    parent_id=a.id,
                    x=cx, y=cy,
                    energy=child_e,
                    age=0,
                    alive=True,
                    genome=child_g
                )
                newborns.append(child)
                self.births_today += 1
                print(f"[night] REPRO parent={a.id} child={child.id} pos=({cx},{cy}) Eparent={a.energy:.1f} Echild={child.energy:.1f}")

        if newborns:
            for c in newborns:
                print(f"[{day_idx * self.cfg.T_DAY}] SPAWN id={c.id} parent={c.parent_id} pos=({c.x},{c.y}) genes={vars(c.genome)}")
            self.agents.extend(newborns)

    # ----- metrics -----

    def record_day_metrics(self):
        self.day_population.append(len(self.agents))
        if len(self.agents) > 0:
            self.day_mean_energy.append(float(np.mean([a.energy for a in self.agents])))
        else:
            self.day_mean_energy.append(0.0)
        self.day_fights.append(self.fights_today)

        # trait distributions snapshot
        if self.agents:
            arr = lambda f: np.array([f(a.genome) for a in self.agents], dtype=float)
            self.trait_series["max_energy"].append(arr(lambda g: g.max_energy))
            self.trait_series["metabolism"].append(arr(lambda g: g.metabolism))
            self.trait_series["move_cost"].append(arr(lambda g: g.move_cost))
            self.trait_series["perception"].append(arr(lambda g: g.perception))
            self.trait_series["strength"].append(arr(lambda g: g.strength))
            self.trait_series["aggression"].append(arr(lambda g: g.aggression))
            self.trait_series["repro_threshold"].append(arr(lambda g: g.repro_threshold))
            self.trait_series["repro_cost"].append(arr(lambda g: g.repro_cost))
        else:
            for k in self.trait_series:
                self.trait_series[k].append(np.array([]))

        # reset daily counters
        self.fights_today = 0
        self.births_today = 0
        self.deaths_today = 0