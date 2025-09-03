from dataclasses import dataclass
from typing import Optional


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
    action: str = "EXPLORE"
