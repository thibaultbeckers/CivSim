"""
Pygame-based visualization package for the civilization simulation.

Provides:
    - COLORS (shared color palette)
    - ChartData (dataclass for charts)
    - PygameMonitor (main visualization class)
"""

from .colors import COLORS
from .chart_data import ChartData
from .monitor import PygameMonitor

__all__ = ["COLORS", "ChartData", "PygameMonitor"]
