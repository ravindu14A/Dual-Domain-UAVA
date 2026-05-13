"""
path.py — Path definition and interpolation.

A path is loaded from a CSV / TXT file with four columns:
    x [m],  y [m],  z [m],  t [s]

The header line is optional (detected automatically).
Use generate_path.py to create sample path files.
"""

import numpy as np
from pathlib import Path as FilePath


class InspectionPath:
    """
    Stores a time-parametrised 3-D trajectory and provides
    position/velocity queries at arbitrary times.
    """

    def __init__(self, filepath: str):
        data = self._load(filepath)
        self._t = data[:, 3]
        self._xyz = data[:, :3]
        if not np.all(np.diff(self._t) > 0):
            raise ValueError("Path time column must be strictly increasing.")

    # ── Loading ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load(filepath: str) -> np.ndarray:
        p = FilePath(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Path file not found: {filepath}")
        raw = p.read_text().strip().splitlines()
        rows = []
        for line in raw:
            parts = line.split(",")
            try:
                rows.append([float(v) for v in parts[:4]])
            except ValueError:
                pass  # skip header / comment lines
        if len(rows) < 2:
            raise ValueError("Path file must contain at least 2 data rows.")
        return np.array(rows)

    # ── Queries ───────────────────────────────────────────────────────────────

    def position_at(self, t: float) -> np.ndarray:
        t = float(np.clip(t, self._t[0], self._t[-1]))
        return np.array([np.interp(t, self._t, self._xyz[:, i]) for i in range(3)])

    def velocity_at(self, t: float, eps: float = 1e-3) -> np.ndarray:
        """Finite-difference velocity estimate [m/s]."""
        t = float(np.clip(t, self._t[0], self._t[-1]))
        t_lo = max(t - eps, self._t[0])
        t_hi = min(t + eps, self._t[-1])
        if t_hi == t_lo:
            return np.zeros(3)
        return (self.position_at(t_hi) - self.position_at(t_lo)) / (t_hi - t_lo)

    @property
    def total_time(self) -> float:
        return float(self._t[-1])

    @property
    def start_time(self) -> float:
        return float(self._t[0])

    @property
    def xyz(self) -> np.ndarray:
        """Full (N, 3) position array for plotting."""
        return self._xyz.copy()

    @property
    def times(self) -> np.ndarray:
        return self._t.copy()
