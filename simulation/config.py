from dataclasses import dataclass, field
from typing import Tuple, Dict


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
