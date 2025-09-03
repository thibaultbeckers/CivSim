# main_pygame.py
# Main simulation file using pygame monitor instead of matplotlib
# All simulation logic remains the same, only the visualization changes

from __future__ import annotations
import math
import time
import dataclasses as dc
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Iterable
import numpy as np

# Import the pygame monitor instead of matplotlib
from pygame_monitor import PygameMonitor

# -------------- Configuration (tweak freely) --------------

@dataclass
class CFG:
    # World
    W: int = 60
    H: int = 40
    SEED: int = 7

    # Timing
    T_DAY: int = 20     # ticks per day
    N_DAYS: int = 500     # total days

    # Carrots
    E_CARROT: float = 20.0
    C_MAX: int = 50
    BASE_SPAWN: float = 0.002        # per free tile per tick (pre-weight)
    W_MID: float = 3.0
    W_HIGH: float = 7.0
    SPAWN_ATTEMPTS_FRAC: float = 0.02  # fraction of tiles to attempt spawns on each tick

    # Genome ranges / defaults (8 genes)
    G_MAX_ENERGY: Tuple[float, float, float] = (60, 140, 100)
    G_METABOLISM: Tuple[float, float, float] = (0.5, 2.0, 1.0)
    G_MOVE_COST: Tuple[float, float, float] = (0.5, 2.0, 1.0)
    G_PERCEPTION: Tuple[int, int, int] = (2, 7, 4)
    G_STRENGTH: Tuple[float, float, float] = (10, 20, 15)
    G_AGGRESSION: Tuple[float, float, float] = (0.0, 1.0, 0.5)
    G_REPRO_THRESH: Tuple[float, float, float] = (50, 100, 70)
    G_REPRO_COST: Tuple[float, float, float] = (15, 40, 30)

    # Mutation (as stdev ~ % of range)
    MUT_PCT: float = 0.1

    # Aging (soft longevity ramp)
    AGE_SOFT: int = 1200
    AGE_EXP_K: float = 1/1000  # effective_metabolism = metabolism * exp(k*(age - AGE_SOFT))

    # Initialization
    N0: int = 50                       # initial agents
    INIT_ENERGY_FRAC: Tuple[float,float] = (0.4, 0.7)

    # Visualization
    R_COLORS: Tuple[str,str,str] = ("#274029", "#4E7D43", "#9BD06D")  # low/mid/high shades (greenish)
    AGENT_CMAP: str = "viridis"         # for energy fill (0..1)
    ACTION_COLORS: Dict[str,str] = field(default_factory=lambda: {
        "SEEK": "#1f77b4",      # blue
        "EXPLORE": "#7f7f7f",   # gray
        "FORAGE": "#2ca02c",    # green
        "COMBAT": "#d62728",    # red
        "IDLE": "#000000",      # black (fallback)
    })
    DOT_SIZE: float = 40.0
    CARROT_SIZE: float = 55.0

    # Rendering / monitor
    RENDER_EVERY: int = 1  # every tick
    UI_FIGSIZE: Tuple[float,float] = (14, 8)
    UI_DPI: int = 110

    # End-of-sim charts
    TRAIT_BINS: int = 20

# -------------- Data structures --------------

@dataclass
class Genome:
    max_energy: float
    metabolism: float
    move_cost: float
    perception: int
    strength: float
    aggression: float
    repro_threshold: float
    repro_cost: float

@dataclass
class Agent:
    id: int
    parent_id: Optional[int]
    x: int
    y: int
    energy: float
    age: int
    alive: bool
    genome: Genome
    action: str = "EXPLORE"  # last known action (for outline color)

# -------------- Utilities --------------

def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(v, lo), hi))

def wrap(x: int, L: int) -> int:
    return int(x % L)

def torus_manhattan(ax, ay, bx, by, W, H) -> int:
    dx = min(abs(ax - bx), W - abs(ax - bx))
    dy = min(abs(ay - by), H - abs(ay - by))
    return dx + dy

# -------------- World: richness (3 organic zones) + carrots --------------

class World:
    def __init__(self, cfg: CFG, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.richness = self._gen_three_zones()  # values {0,1,2} -> low/mid/high
        self.carrots = np.zeros((cfg.H, cfg.W), dtype=bool)

    def _gen_three_zones(self) -> np.ndarray:
        """Generate one high blob, one mid blob; rest low. Soft edges via blur."""
        H, W = self.cfg.H, self.cfg.W
        rng = self.rng

        # Create base noise
        noise = rng.normal(0, 1, size=(H, W))

        # Smooth with simple box blur a few times
        def blur(a: np.ndarray, k: int = 2) -> np.ndarray:
            out = a.copy()
            for _ in range(k):
                # 4-neighbor average with wrap
                out = 0.2 * out + 0.2 * np.roll(out, 1, 0) + 0.2 * np.roll(out, -1, 0) + \
                      0.2 * np.roll(out, 1, 1) + 0.2 * np.roll(out, -1, 1)
            return out

        field = blur(noise, k=3)

        # Choose two seeds (high, mid) and flood-fill-ish mask by thresholding distance on smoothed field
        # Select the top percentile tiles as high, next as mid
        vals = field.flatten()
        hi_thr = np.quantile(vals, 0.92)
        mid_thr = np.quantile(vals, 0.78)

        richness = np.zeros_like(field, dtype=np.uint8)
        richness[field >= hi_thr] = 2
        richness[(field >= mid_thr) & (field < hi_thr)] = 1
        # We want contiguous organic blobs; soften boundaries:
        richness = blur(richness.astype(float), k=1)
        richness = (richness > 1.2).astype(np.uint8) * 2 + (((richness > 0.4) & (richness <= 1.2)).astype(np.uint8) * 1)

        # Guarantee at least one high and one mid tile
        if not np.any(richness == 2):
            ymax, xmax = np.unravel_index(np.argmax(field), field.shape)
            richness[ymax, xmax] = 2
        if not np.any(richness == 1):
            # pick second best
            idx = np.argsort(vals)[-2]
            y2, x2 = np.unravel_index(idx, field.shape)
            richness[y2, x2] = 1

        return richness.astype(np.uint8)

    def spawn_carrots(self):
        """Spawn carrots probabilistically, respecting C_MAX global cap."""
        cfg = self.cfg
        H, W = cfg.H, cfg.W
        current = int(self.carrots.sum())
        capacity = max(0, cfg.C_MAX - current)
        if capacity <= 0:
            return
        free_mask = ~self.carrots
        free_idxs = np.flatnonzero(free_mask)
        if free_idxs.size == 0:
            return

        attempts = min(int(round(cfg.SPAWN_ATTEMPTS_FRAC * H * W)), free_idxs.size, capacity)
        if attempts <= 0:
            return

        # Weights by richness
        rflat = self.richness.flatten()
        weights = np.ones_like(rflat, dtype=float)
        weights[rflat == 1] = cfg.W_MID
        weights[rflat == 2] = cfg.W_HIGH
        # Scale by BASE_SPAWN and mask only free tiles
        p = np.zeros_like(rflat, dtype=float)
        p[free_idxs] = cfg.BASE_SPAWN * weights[free_idxs]
        if p.sum() <= 0:
            return
        p = p / p.sum()

        # Sample tiles without replacement according to p
        chosen = self.rng.choice(np.arange(rflat.size), size=attempts, replace=False, p=p)
        y, x = np.unravel_index(chosen, (H, W))
        self.carrots[y, x] = True

# -------------- Simulation core --------------

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

# -------------- End-of-sim charts (save individual plots instead of showing) --------------

def final_charts(sim: Sim):
    """Generate and save individual analysis charts for the simulation."""
    try:
        import matplotlib.pyplot as plt
        import os
        
        cfg = sim.cfg
        
        # Check if we have any data to plot
        if not sim.day_population:
            print("No simulation data to plot")
            return
        
        # Create output directory
        output_dir = "simulation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        days = np.arange(len(sim.day_population))
        
        # 1. Population over time
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(days, sim.day_population, lw=3, color='#1f77b4')
        plt.title(f"Population Evolution - {len(sim.day_population)} Days", fontsize=16, fontweight='bold')
        plt.xlabel("Day", fontsize=12)
        plt.ylabel("Number of Agents", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/population_evolution.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Fights per day
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(days, sim.day_fights, lw=3, color="#d62728")
        plt.title(f"Combat Frequency - {len(sim.day_population)} Days", fontsize=16, fontweight='bold')
        plt.xlabel("Day", fontsize=12)
        plt.ylabel("Number of Fights", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/combat_frequency.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Mean energy per day
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(days, sim.day_mean_energy, lw=3, color="#2ca02c")
        plt.title(f"Average Energy Levels - {len(sim.day_population)} Days", fontsize=16, fontweight='bold')
        plt.xlabel("Day", fontsize=12)
        plt.ylabel("Mean Energy", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/energy_evolution.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Individual trait evolution plots
        trait_names = {
            "max_energy": "Maximum Energy",
            "metabolism": "Metabolism Rate", 
            "move_cost": "Movement Cost",
            "perception": "Perception Range",
            "strength": "Combat Strength",
            "aggression": "Aggression Level",
            "repro_threshold": "Reproduction Threshold",
            "repro_cost": "Reproduction Cost"
        }
        
        for trait_key, trait_name in trait_names.items():
            if trait_key in sim.trait_series and sim.trait_series[trait_key]:
                plt.figure(figsize=(10, 6), dpi=150)
                
                # Calculate mean for each day
                trait_means = []
                for day_data in sim.trait_series[trait_key]:
                    if day_data.size > 0:
                        trait_means.append(np.mean(day_data))
                    else:
                        trait_means.append(np.nan)
                
                # Plot trait evolution
                plt.plot(days, trait_means, lw=3, color='#9467bd', marker='o', markersize=4)
                plt.title(f"{trait_name} Evolution - {len(sim.day_population)} Days", fontsize=16, fontweight='bold')
                plt.xlabel("Day", fontsize=12)
                plt.ylabel(trait_name, fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save individual trait plot
                filename = f"{output_dir}/{trait_key}_evolution.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
        
        # 5. Summary statistics plot
        plt.figure(figsize=(12, 8), dpi=150)
        
        # Create subplots for key metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
        
        # Population
        ax1.plot(days, sim.day_population, lw=2, color='#1f77b4')
        ax1.set_title("Population", fontweight='bold')
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Agents")
        ax1.grid(True, alpha=0.3)
        
        # Fights
        ax2.plot(days, sim.day_fights, lw=2, color="#d62728")
        ax2.set_title("Daily Fights", fontweight='bold')
        ax2.set_xlabel("Day")
        ax2.set_ylabel("Fights")
        ax2.grid(True, alpha=0.3)
        
        # Energy
        ax3.plot(days, sim.day_mean_energy, lw=2, color="#2ca02c")
        ax3.set_title("Mean Energy", fontweight='bold')
        ax3.set_xlabel("Day")
        ax3.set_ylabel("Energy")
        ax3.grid(True, alpha=0.3)
        
        # Final trait distribution (last day)
        if sim.agents and sim.trait_series["strength"]:
            last_strength = sim.trait_series["strength"][-1]
            if last_strength.size > 0:
                ax4.hist(last_strength, bins=20, alpha=0.7, color='#ff7f0e', edgecolor='black')
                ax4.set_title("Final Strength Distribution", fontweight='bold')
                ax4.set_xlabel("Strength")
                ax4.set_ylabel("Frequency")
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/summary_statistics.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print summary to console
        print(f"\n=== SIMULATION SUMMARY ===")
        print(f"Days completed: {len(sim.day_population)}")
        print(f"Final population: {len(sim.agents) if sim.agents else 0}")
        print(f"Total fights: {sum(sim.day_fights) if sim.day_fights else 0}")
        if sim.agents:
            avg_energy = np.mean([a.energy for a in sim.agents])
            print(f"Average final energy: {avg_energy:.1f}")
        
        print(f"\n📊 Charts saved to '{output_dir}/' directory:")
        print(f"  • population_evolution.png")
        print(f"  • combat_frequency.png") 
        print(f"  • energy_evolution.png")
        print(f"  • [trait]_evolution.png (8 individual trait plots)")
        print(f"  • summary_statistics.png")
        print("========================\n")
        
    except ImportError:
        print("Matplotlib not available for final charts. Install with: pip install matplotlib")
        print(f"\n=== SIMULATION SUMMARY ===")
        print(f"Days completed: {len(sim.day_population)}")
        print(f"Final population: {len(sim.agents) if sim.agents else 0}")
        print(f"Total fights: {sum(sim.day_fights) if sim.day_fights else 0}")
        if sim.agents:
            avg_energy = np.mean([a.energy for a in sim.agents])
            print(f"Average final energy: {avg_energy:.1f}")
        print("========================\n")

# -------------- Main loop --------------

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
                
                # Check stop again after unpausing
                if ui.should_stop:
                    print("Simulation stopped by user")
                    final_charts(sim)
                    return
                
                sim.upkeep()
                if not sim.agents:
                    ui.render(day, t)
                    break  # world died out
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

            # night (instant)
            sim.night_phase(day)
            # record per-day metrics
            sim.record_day_metrics()

        # End-of-sim charts (either completed or stopped)
        if not ui.should_stop:
            print("Simulation completed normally")
        final_charts(sim)
        
    finally:
        # Always cleanup pygame
        ui.cleanup()

if __name__ == "__main__":
    run()
