# civsim_minimal.py
# A small, fun, toroidal-grid civilization sim with discrete carrots, simple combat,
# tiny genome, exponential aging metabolism, live monitor, and end-of-sim charts.

from __future__ import annotations
import math
import time
import dataclasses as dc
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Iterable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import gridspec
from matplotlib.widgets import Button

# -------------- Configuration (tweak freely) --------------

@dataclass
class CFG:
    # World
    W: int = 60
    H: int = 40
    SEED: int = 7

    # Timing
    T_DAY: int = 20     # ticks per day
    N_DAYS: int = 25     # total days

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
    G_STRENGTH: Tuple[float, float, float] = (3, 12, 8)
    G_AGGRESSION: Tuple[float, float, float] = (0.0, 1.0, 0.3)
    G_REPRO_THRESH: Tuple[float, float, float] = (50, 100, 70)
    G_REPRO_COST: Tuple[float, float, float] = (15, 40, 30)

    # Mutation (as stdev ~ % of range)
    MUT_PCT: float = 0.05

    # Aging (soft longevity ramp)
    AGE_SOFT: int = 1200
    AGE_EXP_K: float = 1/2000  # effective_metabolism = metabolism * exp(k*(age - AGE_SOFT))

    # Initialization
    N0: int = 30                       # initial agents
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

# CFG.ACTION_COLORS = CFG.ACTION_COLORS.default_factory()  # ensure dict default

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

        # Age-related deaths (optional hard cap)
        # Here we just log if someone exactly hits ultra-old - but we already starve them via metabolism ramp.
        # If you want a hard cap, uncomment:
        # HARD = self.cfg.AGE_SOFT + 2000
        # still_alive = []
        # for a in self.agents:
        #     if a.age > HARD:
        #         a.alive = False
        #         self.deaths_today += 1
        #         print(f"[night] DEATH id={a.id} cause=AGE pos=({a.x},{a.y}) age={a.age}")
        #     else:
        #         still_alive.append(a)
        # self.agents = still_alive

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

# -------------- Live monitor (matplotlib) --------------

class LiveMonitor:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.cfg = sim.cfg
        self.fig = plt.figure(figsize=self.cfg.UI_FIGSIZE, dpi=self.cfg.UI_DPI, constrained_layout=True)
        gs = gridspec.GridSpec(nrows=3, ncols=4, figure=self.fig, width_ratios=[1.1, 0.65, 0.65, 0.65],
                              height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)

        # Map axes (big)
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_map.set_title("World Map")
        self.ax_map.set_aspect('equal')
        self.ax_map.set_xticks([]); self.ax_map.set_yticks([])

        # Time series
        self.ax_pop = self.fig.add_subplot(gs[0, 1])
        self.ax_energy = self.fig.add_subplot(gs[0, 2])
        self.ax_counts = self.fig.add_subplot(gs[0, 3])  # carrots + fights today

        # Trait histograms (2 rows × 3 cols; we’ll show 6 to keep it simple & fast)
        self.trait_axes = [
            self.fig.add_subplot(gs[1,1]), self.fig.add_subplot(gs[1,2]), self.fig.add_subplot(gs[1,3]),
            self.fig.add_subplot(gs[2,1]), self.fig.add_subplot(gs[2,2]), self.fig.add_subplot(gs[2,3]),
        ]
        self.trait_keys = ["metabolism","move_cost","perception","strength","aggression","repro_threshold"]
        self.trait_ax_titles = ["Metabolism","Move Cost","Perception","Strength","Aggression","Repro Thresh"]

        # Map layers
        # background: richness
        cmap = ListedColormap(self.cfg.R_COLORS)
        self.im_bg = self.ax_map.imshow(self.sim.world.richness, cmap=cmap, vmin=0, vmax=2, origin='upper')
        # carrots
        Y, X = np.nonzero(self.sim.world.carrots)
        self.sc_carrots = self.ax_map.scatter(X, Y, s=self.cfg.CARROT_SIZE, marker="^", facecolors="orange", edgecolors="none", alpha=0.95)
        # agents (init empty)
        self.sc_agents = self.ax_map.scatter([], [], s=self.cfg.DOT_SIZE, marker="o", linewidths=0.8)

        # Add legend for the map
        self._add_map_legend()

        # Add control buttons
        self._add_control_buttons()
        
        # Add keyboard shortcuts
        self._add_keyboard_shortcuts()

        # Time series lines
        self.pop_line, = self.ax_pop.plot([], [], lw=1.5)
        self.ax_pop.set_title("Population")
        self.ax_pop.set_xlabel("Day"); self.ax_pop.set_ylabel("Agents")
        self.energy_line, = self.ax_energy.plot([], [], lw=1.5)
        self.ax_energy.set_title("Mean Energy")
        self.ax_energy.set_xlabel("Day"); self.ax_energy.set_ylabel("Energy")

        # Counts bar (carrots on map & fights today)
        self.ax_counts.set_title("Carrots & Fights (today)")
        self.bar_counts = self.ax_counts.bar(["Carrots", "Fights"], [0,0])

        # Trait histogram artists (initialize)
        self.trait_bars = []
        for ax, title in zip(self.trait_axes, self.trait_ax_titles):
            ax.set_title(title, fontsize=9)
            ax.set_yticks([])
            ax.set_xlabel("")
            bars = ax.bar([0], [0])  # placeholder; real bins set on first update
            self.trait_bars.append((ax, bars, None, None))  # (ax, bars, bin_edges, key)

        plt.pause(0.01)  # allow window to show

    def _add_map_legend(self):
        """Add a legend explaining the map symbols and colors."""
        # Create legend handles for different map elements
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        # Food zones (richness levels)
        zone_handles = [
            Patch(color=self.cfg.R_COLORS[0], label="Low Food Zone"),
            Patch(color=self.cfg.R_COLORS[1], label="Medium Food Zone"),
            Patch(color=self.cfg.R_COLORS[2], label="High Food Zone")
        ]
        
        # Carrots
        carrot_handle = Line2D([0], [0], marker='^', color='orange', markersize=8, 
                              linestyle='', label='Carrots', markeredgecolor='none')
        
        # Agent energy levels (viridis colormap)
        energy_handles = [
            Line2D([0], [0], marker='o', color=plt.get_cmap(self.cfg.AGENT_CMAP)(0.0), 
                   markersize=6, linestyle='', label='Low Energy'),
            Line2D([0], [0], marker='o', color=plt.get_cmap(self.cfg.AGENT_CMAP)(0.5), 
                   markersize=6, linestyle='', label='Medium Energy'),
            Line2D([0], [0], marker='o', color=plt.get_cmap(self.cfg.AGENT_CMAP)(1.0), 
                   markersize=6, linestyle='', label='High Energy')
        ]
        
        # Agent actions (outline colors)
        action_handles = [
            Line2D([0], [0], marker='o', color=self.cfg.ACTION_COLORS["SEEK"], 
                   markersize=6, linestyle='', markeredgecolor=self.cfg.ACTION_COLORS["SEEK"], 
                   markeredgewidth=2, markerfacecolor='none', label='Seeking Food'),
            Line2D([0], [0], marker='o', color=self.cfg.ACTION_COLORS["EXPLORE"], 
                   markersize=6, linestyle='', markeredgecolor=self.cfg.ACTION_COLORS["EXPLORE"], 
                   markeredgewidth=2, markerfacecolor='none', label='Exploring'),
            Line2D([0], [0], marker='o', color=self.cfg.ACTION_COLORS["FORAGE"], 
                   markersize=6, linestyle='', markeredgecolor=self.cfg.ACTION_COLORS["FORAGE"], 
                   markeredgewidth=2, markerfacecolor='none', label='Foraging'),
            Line2D([0], [0], marker='o', color=self.cfg.ACTION_COLORS["COMBAT"], 
                   markersize=6, linestyle='', markeredgecolor=self.cfg.ACTION_COLORS["COMBAT"], 
                   markeredgewidth=2, markerfacecolor='none', label='In Combat')
        ]
        
        # Combine all handles
        all_handles = zone_handles + [carrot_handle] + energy_handles + action_handles
        
        # Add legend below the map (outside the grid)
        self.ax_map.legend(handles=all_handles, loc='upper center', fontsize=8, 
                          title='Map Legend', title_fontsize=9, framealpha=0.8,
                          bbox_to_anchor=(0.5, -0.08), ncol=4)

    def _add_control_buttons(self):
        """Add pause/play and stop buttons to control the simulation."""
        # Create button axes (positioned at the bottom of the figure)
        button_height = 0.06
        button_width = 0.1
        
        # Position buttons below the legend
        # Pause/Play button
        self.ax_play_pause = self.fig.add_axes([0.35, 0.005, button_width, button_height])
        self.btn_play_pause = Button(self.ax_play_pause, '⏸ Pause', color='lightblue', hovercolor='lightcoral')
        self.btn_play_pause.on_clicked(self._toggle_pause)
        
        # Stop button
        self.ax_stop = self.fig.add_axes([0.55, 0.005, button_width, button_height])
        self.btn_stop = Button(self.ax_stop, '⏹ Stop', color='lightcoral', hovercolor='red')
        self.btn_stop.on_clicked(self._stop_simulation)
        
        # Control state
        self.is_paused = False
        self.should_stop = False
        
        # Add status text below the buttons
        self.status_text = self.fig.text(0.5, 0.005, 'Running', ha='center', va='bottom', 
                                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    def _toggle_pause(self, event):
        """Toggle between pause and play states."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_play_pause.label.set_text('▶ Play')
            self.btn_play_pause.color = 'lightgreen'
            self.status_text.set_text('Paused')
            self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        else:
            self.btn_play_pause.label.set_text('⏸ Pause')
            self.btn_play_pause.color = 'lightblue'
            self.status_text.set_text('Running')
            self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    def _stop_simulation(self, event):
        """Stop the simulation and generate final plots."""
        self.should_stop = True
        self.btn_stop.color = 'red'
        self.btn_stop.label.set_text('⏹ Stopped')
        self.status_text.set_text('Stopped - Generating final charts...')
        self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        
        # Disable buttons after stopping
        self.btn_play_pause.disconnect_events()
        self.btn_stop.disconnect_events()

    def _add_keyboard_shortcuts(self):
        """Add keyboard shortcuts for controlling the simulation."""
        def on_key(event):
            if event.key == ' ':  # Spacebar to pause/play
                self._toggle_pause(None)
            elif event.key == 's':  # 's' to stop
                self._stop_simulation(None)
            elif event.key == 'r':  # 'r' to reset (if needed)
                pass  # Could implement reset functionality later
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Add help text above the map
        help_text = "Controls: Space=Pause/Play, S=Stop, R=Reset"
        self.fig.text(0.5, 0.98, help_text, ha='center', va='top', 
                     fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    def _update_map(self):
        # Carrots
        Y, X = np.nonzero(self.sim.world.carrots)
        self.sc_carrots.set_offsets(np.c_[X, Y])

        # Agents
        if self.sim.agents:
            xs = np.array([a.x for a in self.sim.agents])
            ys = np.array([a.y for a in self.sim.agents])
            # fill color by energy fraction
            fracs = np.array([a.energy / max(a.genome.max_energy, 1e-6) for a in self.sim.agents])
            face = plt.get_cmap(self.cfg.AGENT_CMAP)(np.clip(fracs, 0, 1))
            # outline color by action
            act_map = self.cfg.ACTION_COLORS
            edge = [act_map.get(a.action, act_map["IDLE"]) for a in self.sim.agents]
            self.sc_agents.set_offsets(np.c_[xs, ys])
            self.sc_agents.set_facecolors(face)
            self.sc_agents.set_edgecolors(edge)
            self.sc_agents.set_sizes(np.full_like(xs, self.cfg.DOT_SIZE, dtype=float))
        else:
            self.sc_agents.set_offsets(np.empty((0,2)))
        # redraw map extent
        self.ax_map.set_xlim(-0.5, self.cfg.W-0.5)
        self.ax_map.set_ylim(self.cfg.H-0.5, -0.5)

    def _update_timeseries(self, day_idx: int):
        # Population & mean energy per day (only updated at night, so we use last recorded)
        days = np.arange(len(self.sim.day_population))
        self.pop_line.set_data(days, self.sim.day_population)
        self.ax_pop.set_xlim(0, max(1, len(days)))
        self.ax_pop.set_ylim(0, max(1, max(self.sim.day_population) if self.sim.day_population else 1))

        self.energy_line.set_data(days, self.sim.day_mean_energy)
        self.ax_energy.set_xlim(0, max(1, len(days)))
        ymax = max(1, max(self.sim.day_mean_energy) if self.sim.day_mean_energy else 1)
        self.ax_energy.set_ylim(0, ymax * 1.1)

        # Carrots & fights (today) bar
        carrots_now = int(self.sim.world.carrots.sum())
        fights_today = self.sim.fights_today
        self.bar_counts.remove()
        self.bar_counts = self.ax_counts.bar(["Carrots", "Fights"], [carrots_now, fights_today], color=["orange", "#d62728"])

    def _update_trait_hists(self):
        # Snapshot of current distribution for 6 chosen traits
        eps = 1e-9
        agents = self.sim.agents
        if not agents:
            for ax, bars, _, _ in self.trait_bars:
                for b in bars: b.set_height(0)
            return

        # Build data
        cur = {
            "metabolism": np.array([a.genome.metabolism for a in agents]),
            "move_cost": np.array([a.genome.move_cost for a in agents]),
            "perception": np.array([a.genome.perception for a in agents]),
            "strength": np.array([a.genome.strength for a in agents]),
            "aggression": np.array([a.genome.aggression for a in agents]),
            "repro_threshold": np.array([a.genome.repro_threshold for a in agents]),
        }
        ranges = {
            "metabolism": self.sim.cfg.G_METABOLISM[:2],
            "move_cost": self.sim.cfg.G_MOVE_COST[:2],
            "perception": self.sim.cfg.G_PERCEPTION[:2],
            "strength": self.sim.cfg.G_STRENGTH[:2],
            "aggression": self.sim.cfg.G_AGGRESSION[:2],
            "repro_threshold": self.sim.cfg.G_REPRO_THRESH[:2],
        }

        for (ax, bars, bin_edges, key), k in zip(self.trait_bars, self.trait_keys):
            data = cur[k]
            lo, hi = ranges[k]
            bins = max(6, min(16, self.sim.cfg.TRAIT_BINS))
            hist, edges = np.histogram(data, bins=bins, range=(lo, hi))
            centers = 0.5 * (edges[:-1] + edges[1:])
            if bin_edges is None:
                # first time: create bars
                ax.clear()
                ax.set_title(ax.get_title())
                ax.set_yticks([])
                ax.bar(centers, hist, width=(edges[1]-edges[0]) * 0.9, color="#5DADE2")
                ax.set_xlim(lo, hi)
                # Add axis labels for the distribution graphs
                ax.set_xlabel(k.replace("_", " ").title())
                ax.set_ylabel("Count")
                self.trait_bars[self.trait_axes.index(ax)] = (ax, ax.patches, edges, k)
            else:
                # update heights
                for b, h in zip(ax.patches, hist):
                    b.set_height(h)

    def render(self, day_idx: int, tick_idx: int):
        self._update_map()
        # timeseries uses per-day metrics; still refresh so user sees latest carrots/fights today
        self._update_timeseries(day_idx)
        self._update_trait_hists()
        self.fig.suptitle(f"Day {day_idx+1}/{self.cfg.N_DAYS} — Tick {tick_idx+1}/{self.cfg.T_DAY} — Pop {len(self.sim.agents)}", fontsize=12)
        
        # Update status
        self.update_status(day_idx, tick_idx)
        
        plt.pause(0.001)  # render every tick

    def should_continue(self) -> bool:
        """Check if the simulation should continue running."""
        return not self.should_stop

    def update_status(self, day_idx: int, tick_idx: int):
        """Update the status text with current simulation progress."""
        if self.should_stop:
            self.status_text.set_text(f'Stopped at Day {day_idx+1}, Tick {tick_idx+1}')
        elif self.is_paused:
            self.status_text.set_text(f'Paused at Day {day_idx+1}, Tick {tick_idx+1}')
        else:
            self.status_text.set_text(f'Running - Day {day_idx+1}, Tick {tick_idx+1}')

    def get_simulation_summary(self) -> str:
        """Get a summary of the current simulation state."""
        if not hasattr(self.sim, 'day_population') or not self.sim.day_population:
            return "No simulation data available"
        
        days_completed = len(self.sim.day_population)
        current_pop = len(self.sim.agents) if self.sim.agents else 0
        total_fights = sum(self.sim.day_fights) if self.sim.day_fights else 0
        
        return f"Days: {days_completed}, Current Population: {current_pop}, Total Fights: {total_fights}"

# -------------- End-of-sim charts --------------

def final_charts(sim: Sim):
    """Generate final analysis charts for the simulation."""
    cfg = sim.cfg
    
    # Check if we have any data to plot
    if not sim.day_population:
        print("No simulation data to plot")
        return
    
    days = np.arange(len(sim.day_population))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=cfg.UI_DPI, constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Population over time
    ax1.plot(days, sim.day_population, lw=2)
    ax1.set_title("Population over time"); ax1.set_xlabel("Day"); ax1.set_ylabel("Agents")

    # Fights per day
    ax2.plot(days, sim.day_fights, lw=2, color="#d62728")
    ax2.set_title("Fights per day"); ax2.set_xlabel("Day"); ax2.set_ylabel("# Fights")

    # Mean energy per day
    ax3.plot(days, sim.day_mean_energy, lw=2, color="#2ca02c")
    ax3.set_title("Mean energy over time"); ax3.set_xlabel("Day"); ax3.set_ylabel("Energy")

    # Trait means over time (simple & readable)
    trait_means = {}
    for k, series in sim.trait_series.items():
        # mean per day (skip empty)
        m = [np.mean(s) if s.size > 0 else np.nan for s in series]
        trait_means[k] = m

    for k, m in trait_means.items():
        ax4.plot(days, m, label=k.replace("_"," "))
    ax4.set_title("Trait means over time")
    ax4.set_xlabel("Day"); ax4.set_ylabel("Value"); ax4.legend(fontsize=8, ncols=2)

    # Add more detailed information
    if hasattr(sim, 'day_population') and sim.day_population:
        total_fights = sum(sim.day_fights) if sim.day_fights else 0
        final_pop = len(sim.agents) if sim.agents else 0
        title = f"Simulation Results - {len(sim.day_population)} days completed\nTotal Fights: {total_fights}, Final Population: {final_pop}"
    else:
        title = "Simulation Results"
    
    plt.suptitle(title, fontsize=14)
    plt.show()
    
    # Print summary to console
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
    ui = LiveMonitor(sim)

    ticks_total = cfg.N_DAYS * cfg.T_DAY
    tcount = 0
    
    for day in range(cfg.N_DAYS):
        for t in range(cfg.T_DAY):
            # Check if simulation should stop
            if ui.should_stop:
                print("Simulation stopped by user")
                final_charts(sim)
                return
            
            # Check if simulation is paused
            while ui.is_paused and not ui.should_stop:
                ui.render(day, t)
                plt.pause(0.1)  # Small pause to allow button clicks
            
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
                ui.render(day, t)
            tcount += 1

        # night (instant)
        sim.night_phase(day)
        # record per-day metrics
        sim.record_day_metrics()

    # End-of-sim charts (either completed or stopped)
    if not ui.should_stop:
        print("Simulation completed normally")
    final_charts(sim)

if __name__ == "__main__":
    run()
