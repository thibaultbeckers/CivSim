from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ChartData:
    """Data structure for chart rendering"""
    values: List[float]
    max_value: float
    min_value: float
    color: Tuple[int, int, int]
    title: str
