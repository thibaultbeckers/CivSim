def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(v, lo), hi))


def wrap(x: int, L: int) -> int:
    return int(x % L)


def torus_manhattan(ax, ay, bx, by, W, H) -> int:
    dx = min(abs(ax - bx), W - abs(ax - bx))
    dy = min(abs(ay - by), H - abs(ay - by))
    return dx + dy
