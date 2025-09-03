import numpy as np
from simulation.config import CFG

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