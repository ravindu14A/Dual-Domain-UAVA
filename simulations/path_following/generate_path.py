"""
generate_path.py — Utility to create path CSV files for the drone simulation.

Each output CSV has one header row and then data rows:
    x [m], y [m], z [m], t [s]

Run examples
------------
python generate_path.py --type helix     --out sample_helix.csv
python generate_path.py --type figure8   --out sample_figure8.csv
python generate_path.py --type inspection --out sample_inspection.csv
python generate_path.py --type waypoints --out sample_wps.csv
"""

import argparse
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Path generators  (each returns (N,4) array: x, y, z, t)
# ─────────────────────────────────────────────────────────────────────────────

def helix_path(radius: float = 20.0,
               z_start: float = 10.0,
               z_end:   float = 40.0,
               n_turns: float = 2.0,
               duration: float = 60.0,
               dt: float = 0.1) -> np.ndarray:
    """Rising helix."""
    t = np.arange(0.0, duration + dt, dt)
    angle = 2.0 * np.pi * n_turns * (t / duration)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = np.linspace(z_start, z_end, len(t))
    return np.column_stack([x, y, z, t])


def figure8_path(size: float = 20.0,
                 altitude: float = 20.0,
                 duration: float = 60.0,
                 dt: float = 0.1) -> np.ndarray:
    """Horizontal figure-of-eight (lemniscate of Bernoulli)."""
    t = np.arange(0.0, duration + dt, dt)
    s = 2.0 * np.pi * (t / duration)
    denom = 1.0 + np.sin(s) ** 2
    x = size * np.cos(s) / denom
    y = size * np.sin(s) * np.cos(s) / denom
    z = np.full_like(t, altitude)
    return np.column_stack([x, y, z, t])


def circle_path(radius: float = 15.0,
                altitude: float = 20.0,
                period: float = 30.0,
                n_laps: float = 2.0,
                dt: float = 0.1) -> np.ndarray:
    """Horizontal circle."""
    duration = period * n_laps
    t = np.arange(0.0, duration + dt, dt)
    angle = 2.0 * np.pi * (t / period)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = np.full_like(t, altitude)
    return np.column_stack([x, y, z, t])


def inspection_path(tower_radius: float = 4.0,
                    z_bottom: float = 5.0,
                    z_top:    float = 30.0,
                    n_strips: int = 4,
                    v_scan:   float = 2.0,
                    dt:       float = 0.1) -> np.ndarray:
    """
    Simplified lawnmower around a vertical cylinder (tower inspection).
    Alternating ascending / descending vertical strips at the tower radius.
    """
    strip_h = z_top - z_bottom
    strip_t = strip_h / v_scan            # time per vertical strip
    trans_t = (2.0 * np.pi * tower_radius / n_strips) / v_scan  # horizontal hop

    rows = []
    t_now = 0.0
    theta_now = 0.0
    d_theta = 2.0 * np.pi / n_strips

    for i in range(n_strips):
        ascending = (i % 2 == 0)
        z_a = z_bottom if ascending else z_top
        z_b = z_top    if ascending else z_bottom
        n_pts = max(int(strip_t / dt), 2)
        for j in range(n_pts):
            frac = j / (n_pts - 1)
            z = z_a + (z_b - z_a) * frac
            x = tower_radius * np.cos(theta_now)
            y = tower_radius * np.sin(theta_now)
            rows.append([x, y, z, t_now + frac * strip_t])
        t_now += strip_t

        if i < n_strips - 1:
            theta_end = theta_now + d_theta
            n_hop = max(int(trans_t / dt), 2)
            z_hop = z_b
            for j in range(1, n_hop + 1):
                frac = j / n_hop
                th = theta_now + d_theta * frac
                x = tower_radius * np.cos(th)
                y = tower_radius * np.sin(th)
                rows.append([x, y, z_hop, t_now + frac * trans_t])
            theta_now = theta_end
            t_now += trans_t

    return np.array(rows)


def waypoints_path(waypoints: list,
                   speed: float = 3.0,
                   dt: float = 0.1) -> np.ndarray:
    """
    Smooth path through a list of (x, y, z) waypoints at constant speed.
    Simple linear interpolation; caller can replace with spline for smoothness.
    """
    wps = np.array(waypoints, dtype=float)
    rows = []
    t_now = 0.0

    for i in range(len(wps) - 1):
        seg = wps[i + 1] - wps[i]
        dist = float(np.linalg.norm(seg))
        if dist < 1e-6:
            continue
        seg_t = dist / speed
        n_pts = max(int(seg_t / dt), 2)
        for j in range(n_pts):
            frac = j / n_pts
            p = wps[i] + seg * frac
            rows.append([p[0], p[1], p[2], t_now + frac * seg_t])
        t_now += seg_t

    p = wps[-1]
    rows.append([p[0], p[1], p[2], t_now])
    return np.array(rows)


def default_waypoints():
    return [
        [  0,  0, 10],
        [ 30,  0, 15],
        [ 30, 30, 20],
        [  0, 30, 15],
        [  0,  0, 10],
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(data: np.ndarray, filepath: str):
    Path(filepath).write_text(
        "x,y,z,t\n" +
        "\n".join(f"{r[0]:.4f},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f}" for r in data)
    )
    print(f"Saved {len(data)} points → {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate drone path CSV files")
    parser.add_argument("--type", choices=["helix", "figure8", "circle",
                                            "inspection", "waypoints"],
                        default="helix")
    parser.add_argument("--out", default=None, help="Output CSV file path")
    args = parser.parse_args()

    generators = {
        "helix":      lambda: helix_path(),
        "figure8":    lambda: figure8_path(),
        "circle":     lambda: circle_path(),
        "inspection": lambda: inspection_path(),
        "waypoints":  lambda: waypoints_path(default_waypoints()),
    }

    data = generators[args.type]()
    out  = args.out or f"./sample_{args.type}.csv"
    save_csv(data, out)


if __name__ == "__main__":
    main()
