"""
D_sweep.py  —  Sweep standoff distance D and plot total inspection distance and
               mission time for lawnmower vs spiral across the full inspection path.

Full inspection path (in order):
  1. Underwater monopile     R=4 m,   H=60 m   (cylinder)
  2. Air cylinder            R=4 m,   H=30 m   (cylinder)
  3. Air cone                R=4→2.85 m, H=135 m (tapered, slant-height used)
  4. Turbine blades          3 × R=3.5 m, H=110 m (cylinders)

Transitions included:
  - Underwater → Air cylinder : vertical ascent H_water at v_max_uw
  - Air cylinder → Cone       : zero (continuous on same structure)
  - Cone → first blade        : one full circumference at R_top+D, speed v_horiz_air
  - Between blades (×2)       : 120° arc at R_blade+D, speed v_horiz_turb

Camera: each phase uses the camera_type from its own config in Test_time.py.
        All other camera params come from Test_time; only D is swept uniformly.
        Spiral is disabled for the whole sweep if any phase uses HYPERSPECTRAL.
"""

import contextlib
import io
import numpy as np
import matplotlib.pyplot as plt

from Test_time import (
    R_base, R_top,
    H_water, H_air_cyl, H_air_cone,
    R_blade, H_blade,
    water_config, air_config, turbine_config,
    cameras,
)
from Test_time_functions import (
    get_w_arc, get_v_frame,
    get_distance_lawnmower, get_distance_spiral,
    get_velocity_rgb_lawnmower, get_velocity_rgb_spiral,
    get_velocity_hyper_lawnmower,
    get_velocity_event, get_velocity_event_spiral,
    time_lawnmower, time_spiral,
)

# ── 1. Per-phase camera setup ─────────────────────────────────────────────────
cam_type_water = water_config["camera_type"]
cam_type_air   = air_config["camera_type"]
cam_type_turb  = turbine_config["camera_type"]

cam_base_water = cameras[cam_type_water]
cam_base_air   = cameras[cam_type_air]
cam_base_turb  = cameras[cam_type_turb]

def _phase_spiral_ok(cam_type, cam_base):
    return not (cam_type == "HYPERSPECTRAL" and "v_fov" not in cam_base)

_SPIRAL_OK = (
    _phase_spiral_ok(cam_type_water, cam_base_water) and
    _phase_spiral_ok(cam_type_air,   cam_base_air)   and
    _phase_spiral_ok(cam_type_turb,  cam_base_turb)
)

print(
    f"[D_sweep] Water: {cam_type_water} | Air: {cam_type_air} | Turbine: {cam_type_turb}"
    f" | Spiral enabled: {_SPIRAL_OK}"
)

# ── 2. Phase speed parameters ─────────────────────────────────────────────────
v_max_uw     = float(water_config["v_max"])
v_horiz_uw   = float(water_config.get("v_horiz", 0.2))
v_max_air    = float(air_config["v_max"])
v_horiz_air  = float(air_config.get("v_horiz", 0.2))
v_max_turb   = float(turbine_config["v_max"])
v_horiz_turb = float(turbine_config.get("v_horiz", 0.2))

# ── 3. Per-section helper ─────────────────────────────────────────────────────

def _v_scan(cam, cam_type, D, R, v_max, mode):
    """Return scan speed [m/s] for the given camera, standoff, and mode."""
    with contextlib.redirect_stdout(io.StringIO()):
        w_arc = get_w_arc(R, D, cam["h_fov"])

    if cam_type == "RGB":
        v_frame = get_v_frame(D, cam["v_fov"])
        if mode == "lawnmower":
            v, _ = get_velocity_rgb_lawnmower(
                v_max, cam["gsd"], cam["max_blur"], cam["shutter"],
                v_frame, cam["v_overlap"], cam["fps"])
        else:
            v, _ = get_velocity_rgb_spiral(
                v_max, R, v_frame, w_arc,
                cam["gsd"], cam["max_blur"], cam["shutter"],
                cam["h_overlap"], cam["fps"])

    elif cam_type == "HYPERSPECTRAL":
        if mode == "spiral":
            raise ValueError("Spiral not supported for HYPERSPECTRAL camera.")
        v, _ = get_velocity_hyper_lawnmower(
            v_max, cam["gsd"], cam["line_rate"],
            cam.get("aintegration", 0.0), cam["max_blur"])

    elif cam_type == "EVENT":
        if mode == "lawnmower":
            v, _ = get_velocity_event(v_max)
        else:
            v, _ = get_velocity_event_spiral(v_max)

    else:
        raise ValueError(f"Unknown camera type: {cam_type}")

    return v


def section(R_s, R_t, H_s, D, cam, cam_type, v_max, v_horiz, mode, n=1):
    """
    Distance [m] and time [s] for one inspection section.

    R_s, R_t : start / end radius (equal for cylinders)
    H_s      : section height [m]
    D        : standoff [m]
    n        : number of identical copies (e.g. 3 blades)
    """
    with contextlib.redirect_stdout(io.StringIO()):
        w_arc = get_w_arc(R_s, D, cam["h_fov"])

    if mode == "lawnmower":
        dists   = get_distance_lawnmower(R_s, R_t, H_s, w_arc)
        d_vert  = dists["vertical"]
        d_horiz = dists["horizontal"]
        v       = _v_scan(cam, cam_type, D, R_s, v_max, "lawnmower")
        d       = d_vert + d_horiz
        t       = time_lawnmower(d_vert, d_horiz, v, v_horiz)
    else:
        v_frame = get_v_frame(D, cam["v_fov"])
        d       = get_distance_spiral(R_s, R_t, H_s, v_frame)
        v       = _v_scan(cam, cam_type, D, R_s, v_max, "spiral")
        t       = time_spiral(d, v)

    return n * d, n * t


# ── 4. Sweep ──────────────────────────────────────────────────────────────────
D_arr = np.linspace(0.5, 10.0, 60)

dist_lm = np.zeros(len(D_arr))
dist_sp = np.zeros(len(D_arr))
time_lm = np.zeros(len(D_arr))
time_sp = np.zeros(len(D_arr))

for i, D in enumerate(D_arr):
    # Each phase uses its own camera base; only D is swept uniformly across all
    cam_w = {**cam_base_water, "D": D}
    cam_a = {**cam_base_air,   "D": D}
    cam_t = {**cam_base_turb,  "D": D}

    # ── section 1: underwater ──
    d1_lm, t1_lm = section(R_base, R_base, H_water, D, cam_w,
                            cam_type_water, v_max_uw, v_horiz_uw, "lawnmower")
    if _SPIRAL_OK:
        d1_sp, t1_sp = section(R_base, R_base, H_water, D, cam_w,
                                cam_type_water, v_max_uw, v_horiz_uw, "spiral")

    # transition: underwater → air  (vertical ascent at v_max_uw)
    d_tr1 = float(H_water)
    t_tr1 = d_tr1 / v_max_uw

    # ── section 2: air cylinder ──
    d2_lm, t2_lm = section(R_base, R_base, H_air_cyl, D, cam_a,
                            cam_type_air, v_max_air, v_horiz_air, "lawnmower")
    if _SPIRAL_OK:
        d2_sp, t2_sp = section(R_base, R_base, H_air_cyl, D, cam_a,
                                cam_type_air, v_max_air, v_horiz_air, "spiral")

    # transition: air cylinder → cone: zero (continuous)
    d_tr2, t_tr2 = 0.0, 0.0

    # ── section 3: air cone ──
    d3_lm, t3_lm = section(R_base, R_top, H_air_cone, D, cam_a,
                            cam_type_air, v_max_air, v_horiz_air, "lawnmower")
    if _SPIRAL_OK:
        d3_sp, t3_sp = section(R_base, R_top, H_air_cone, D, cam_a,
                                cam_type_air, v_max_air, v_horiz_air, "spiral")

    # transition: cone top → first blade (one full circumference at R_top+D)
    d_tr3 = 2.0 * np.pi * (R_top + D)
    t_tr3 = d_tr3 / v_horiz_air

    # ── section 4: turbine blades (3 identical, 120° apart) ──
    d4_lm, t4_lm = section(R_blade, R_blade, H_blade, D, cam_t,
                            cam_type_turb, v_max_turb, v_horiz_turb, "lawnmower", n=3)
    if _SPIRAL_OK:
        d4_sp, t4_sp = section(R_blade, R_blade, H_blade, D, cam_t,
                                cam_type_turb, v_max_turb, v_horiz_turb, "spiral", n=3)

    # transition: between blades (2 moves, each 120° arc at R_blade+D)
    d_tr4 = 2.0 * (2.0 * np.pi * (R_blade + D) / 3.0)
    t_tr4 = d_tr4 / v_horiz_turb

    # ── totals ──
    d_trans = d_tr1 + d_tr2 + d_tr3 + d_tr4
    t_trans = t_tr1 + t_tr2 + t_tr3 + t_tr4

    dist_lm[i] = d1_lm + d2_lm + d3_lm + d4_lm + d_trans
    time_lm[i] = t1_lm + t2_lm + t3_lm + t4_lm + t_trans

    if _SPIRAL_OK:
        dist_sp[i] = d1_sp + d2_sp + d3_sp + d4_sp + d_trans
        time_sp[i] = t1_sp + t2_sp + t3_sp + t4_sp + t_trans

# ── 5. Plots ──────────────────────────────────────────────────────────────────
_STYLE = {"lawnmower": dict(color="#4fc3f7", lw=2.0, label="Lawnmower"),
          "spiral":    dict(color="#ef9a9a", lw=2.0, label="Spiral")}

# Build a camera label for the title showing unique types per phase
_cam_label_parts = [f"Water:{cam_type_water}", f"Air:{cam_type_air}", f"Turbine:{cam_type_turb}"]
_cam_label = "  |  ".join(_cam_label_parts)

# Vertical reference lines for the D used in each phase config
_D_refs = {
    "Water":   cam_base_water["D"],
    "Air":     cam_base_air["D"],
    "Turbine": cam_base_turb["D"],
}
_ref_colors = {"Water": "#80cbc4", "Air": "#ffe082", "Turbine": "#ce93d8"}

def _add_D_refs(ax):
    shown = {}
    for phase, D_ref in _D_refs.items():
        label = f"{phase} D={D_ref} m" if D_ref not in shown.values() else f"_{phase}"
        ax.axvline(D_ref, color=_ref_colors[phase], lw=1.0, ls="--", alpha=0.7, label=label)
        shown[phase] = D_ref

fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(D_arr, dist_lm / 1e3, **_STYLE["lawnmower"])
if _SPIRAL_OK:
    ax1.plot(D_arr, dist_sp / 1e3, **_STYLE["spiral"])
_add_D_refs(ax1)
ax1.set_xlabel("Standoff distance D  [m]", fontsize=11)
ax1.set_ylabel("Total path distance  [km]", fontsize=11)
ax1.set_title(f"Total inspection distance vs standoff\n({_cam_label})", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)
ax1.set_xlim(D_arr[0], D_arr[-1])
fig1.tight_layout()
fig1.savefig("D_sweep_distance.png", dpi=150)

fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(D_arr, time_lm / 3600, **_STYLE["lawnmower"])
if _SPIRAL_OK:
    ax2.plot(D_arr, time_sp / 3600, **_STYLE["spiral"])
_add_D_refs(ax2)
ax2.set_xlabel("Standoff distance D  [m]", fontsize=11)
ax2.set_ylabel("Total mission time  [hours]", fontsize=11)
ax2.set_title(f"Total mission time vs standoff\n({_cam_label})", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25)
ax2.set_xlim(D_arr[0], D_arr[-1])
fig2.tight_layout()
fig2.savefig("D_sweep_time.png", dpi=150)

plt.show()
print("[D_sweep] Done. Figures saved: D_sweep_distance.png, D_sweep_time.png")
