"""
main.py — Drone path-following simulation with 3-D animation.

Usage
-----
  # Run with the default sample path and 6 rotors (from config):
  python main.py

  # Custom path file and rotor count:
  python main.py --path sample_helix.csv --rotors 4

  # Save animation to file instead of displaying:
  python main.py --save

Outputs
-------
  results/  tracking_<N>rotors.png   — time-series position error + power
            animation_<N>rotors.mp4  — (if --save or save_animation: true in config)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider
import json

# ── Local imports (same package directory) ────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from path       import InspectionPath
from drone      import DroneModel
from controller import PathFollowController
from generate_path import helix_path, save_csv

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH  = _HERE / "config.yaml"
RESULTS_DIR  = _HERE / "results"

ROTOR_COLORS = {4: "#E91E63", 6: "#2196F3", 8: "#4CAF50"}


# =============================================================================
# Configuration
# =============================================================================

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Simulation loop  (pure numerics, no plotting)
# =============================================================================

def run_simulation(config: dict, path: InspectionPath,
                   n_rotors: int) -> dict:
    """
    Simulate the drone following *path* with *n_rotors* rotors.

    Returns a dict of arrays (one element per timestep) plus metadata.
    """
    sim_cfg = config.get("simulation", {})
    dt      = float(sim_cfg.get("dt", 0.02))

    # Start drone directly below the path's first point
    p0  = path.position_at(path.start_time)
    pos0 = (p0[0], p0[1], max(p0[2] - 5.0, 0.5))  # slightly below, snap up

    drone = DroneModel(config, n_rotors, initial_pos=pos0)
    ctrl  = PathFollowController(config, drone)

    # Initialise drone with hover thrust so it doesn't fall immediately
    drone.set_rotor_thrusts(
        np.full(n_rotors, drone.hover_thrust / n_rotors)
    )

    t_end = path.total_time
    t     = path.start_time

    # Storage lists
    t_l, pos_l, target_l = [], [], []
    err_l, bat_l, pwr_l  = [], [], []
    att_l, thr_l          = [], []
    vel_l, rates_l        = [], []

    while t <= t_end + dt * 0.5 and drone.battery > 0.0:
        target = path.position_at(t)

        # Controller computes per-rotor thrusts
        thrusts = ctrl.compute(target, drone.state, dt)
        drone.set_rotor_thrusts(thrusts)

        # Physics step
        drone.step(dt)
        drone.drain_battery(dt)

        # Record
        err = float(np.linalg.norm(drone.position - target))
        t_l.append(t)
        pos_l.append(drone.position.copy())
        vel_l.append(drone.velocity.copy())
        target_l.append(target.copy())
        err_l.append(err)
        bat_l.append(drone.battery)
        pwr_l.append(drone.rotor_power())
        att_l.append(drone.attitude.copy())
        rates_l.append(drone.body_rates.copy())
        thr_l.append(drone.rotor_thrusts.copy())

        t += dt

    t_arr = np.array(t_l)
    return {
        "t":        t_arr,
        "pos":      np.array(pos_l),
        "vel":      np.array(vel_l),
        "target":   np.array(target_l),
        "error":    np.array(err_l),
        "battery":  np.array(bat_l),
        "power":    np.array(pwr_l),
        "attitude": np.array(att_l),
        "rates":    np.array(rates_l),
        "thrusts":  np.array(thr_l),
        "n_rotors": n_rotors,
        "bat_cap":  drone.bat_cap,
        "drone":    drone,
        "dt":       dt,
    }


# =============================================================================
# Kinematics export  (CSV + JSON for Unity / Blender)
# =============================================================================

def _euler_to_quat(roll: np.ndarray, pitch: np.ndarray,
                   yaw: np.ndarray):
    """ZYX Euler → unit quaternion (w, x, y, z).  All inputs in radians."""
    cr, sr = np.cos(roll  / 2), np.sin(roll  / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw   / 2), np.sin(yaw   / 2)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return qw, qx, qy, qz


def export_kinematics(result: dict, out_dir: Path) -> None:
    """
    Export full drone kinematics to CSV and JSON.

    Coordinate system (both files)
    ───────────────────────────────
      World frame : right-hand, Z-up
        x = East, y = North, z = Up
      Body frame  : x = forward, y = left, z = up
      Euler angles: ZYX convention (yaw applied first, then pitch, then roll)
      Quaternion  : (w, x, y, z) — standard Hamilton convention

    Unity import notes
    ──────────────────
      Unity is left-hand, Y-up.
        position  → Unity(x, z, y)       [swap y↔z]
        rotation  → Quaternion(-qx, -qz, -qy, qw)  [negate imaginary parts,
                                                      swap qy↔qz, Unity xyzw]
      Import the CSV via a C# MonoBehaviour that reads lines and sets
      transform.position / transform.rotation each FixedUpdate tick.

    Blender import notes
    ────────────────────
      Blender default is right-hand, Z-up — same as this data.
        position  → object.location = (x, y, z)   [direct]
        rotation  → object.rotation_mode = 'QUATERNION'
                    object.rotation_quaternion = (qw, qx, qy, qz)
      Use the companion blender_import.py script (generated alongside this file).
    """
    n   = result["n_rotors"]
    t   = result["t"]
    pos = result["pos"]
    vel = result["vel"]
    att = result["attitude"]
    rates = result["rates"]
    thr = result["thrusts"]
    pwr = result["power"]
    bat = result["battery"]

    roll, pitch, yaw = att[:, 0], att[:, 1], att[:, 2]
    qw, qx, qy, qz   = _euler_to_quat(roll, pitch, yaw)

    drone = result["drone"]

    # Per-rotor power  P_i = T_i^1.5 / sqrt(ρ·A·η_FM) · (1+k0) / η_m
    rho, A  = drone.rho, drone.A_rotor
    eta_fm  = drone.eta_fm
    eta_m   = drone.eta_m
    k0      = drone.k0
    T_tot   = thr.sum(axis=1)

    P_rotor = np.zeros_like(thr)
    for i in range(n):
        Ti = np.maximum(thr[:, i], 0.01)
        P_rotor[:, i] = (Ti**1.5 / np.sqrt(rho * A * eta_fm)) * (1 + k0) / eta_m

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    rotor_T_cols = ",".join(f"T_{i+1}_N" for i in range(n))
    rotor_P_cols = ",".join(f"P_{i+1}_W" for i in range(n))
    header = (
        "# Drone kinematics export\n"
        f"# Rotors: {n}  |  dt: {result['dt']:.4f} s\n"
        "# World frame: right-hand Z-up (x=East, y=North, z=Up)\n"
        "# Euler: ZYX (yaw-pitch-roll)  |  Quaternion: w,x,y,z (Hamilton)\n"
        "# Unity: pos→(x,z,y)  rot→Quaternion(-qx,-qz,-qy,qw)  [xyzw order]\n"
        "# Blender: pos→(x,y,z) direct  rot_quaternion→(qw,qx,qy,qz)\n"
        "#\n"
        f"time_s,x_m,y_m,z_m,vx_ms,vy_ms,vz_ms,"
        f"roll_rad,pitch_rad,yaw_rad,qw,qx,qy,qz,"
        f"p_rads,q_rads,r_rads,"
        f"T_total_N,{rotor_T_cols},"
        f"P_total_W,{rotor_P_cols},"
        f"battery_Wh"
    )

    rows = []
    for k in range(len(t)):
        row = [
            f"{t[k]:.4f}",
            f"{pos[k,0]:.4f}", f"{pos[k,1]:.4f}", f"{pos[k,2]:.4f}",
            f"{vel[k,0]:.4f}", f"{vel[k,1]:.4f}", f"{vel[k,2]:.4f}",
            f"{roll[k]:.6f}",  f"{pitch[k]:.6f}", f"{yaw[k]:.6f}",
            f"{qw[k]:.6f}",    f"{qx[k]:.6f}",    f"{qy[k]:.6f}", f"{qz[k]:.6f}",
            f"{rates[k,0]:.6f}", f"{rates[k,1]:.6f}", f"{rates[k,2]:.6f}",
            f"{T_tot[k]:.4f}",
        ] + [f"{thr[k,i]:.4f}" for i in range(n)] + [
            f"{pwr[k]:.2f}",
        ] + [f"{P_rotor[k,i]:.2f}" for i in range(n)] + [
            f"{bat[k]:.4f}",
        ]
        rows.append(",".join(row))

    csv_path = out_dir / f"kinematics_{n}rotors.csv"
    csv_path.write_text(header + "\n" + "\n".join(rows), encoding="utf-8")
    print(f"  Saved {csv_path.name}  ({len(rows)} rows)")

    # ── JSON (Unity-friendly structured format) ────────────────────────────────
    payload = {
        "meta": {
            "n_rotors": n,
            "dt_s": result["dt"],
            "n_frames": len(t),
            "coordinate_system": "right_hand_z_up",
            "axes": {"x": "East", "y": "North", "z": "Up"},
            "euler_convention": "ZYX_rad",
            "quaternion": "w_x_y_z_Hamilton",
            "unity": {
                "position":   "swap_yz: (x, z, y)",
                "quaternion": "negate_xyz_swap_yz: (-qx, -qz, -qy, qw)  [Unity xyzw]",
            },
            "blender": {
                "position":   "direct (x, y, z)",
                "quaternion": "rotation_quaternion = (qw, qx, qy, qz)",
            },
        },
        "frames": [
            {
                "t":   float(t[k]),
                "pos": [float(pos[k,0]), float(pos[k,1]), float(pos[k,2])],
                "vel": [float(vel[k,0]), float(vel[k,1]), float(vel[k,2])],
                "euler_zyx": [float(roll[k]), float(pitch[k]), float(yaw[k])],
                "quat_wxyz": [float(qw[k]), float(qx[k]), float(qy[k]), float(qz[k])],
                "body_rates": [float(rates[k,0]), float(rates[k,1]), float(rates[k,2])],
                "T_total_N": float(T_tot[k]),
                "T_rotors_N": [float(thr[k,i]) for i in range(n)],
                "P_total_W": float(pwr[k]),
                "P_rotors_W": [float(P_rotor[k,i]) for i in range(n)],
                "battery_Wh": float(bat[k]),
            }
            for k in range(len(t))
        ],
    }

    json_path = out_dir / f"kinematics_{n}rotors.json"
    json_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"  Saved {json_path.name}")

    # ── Blender Python import helper ───────────────────────────────────────────
    blender_script = f'''\
"""
blender_import_{n}rotors.py  — run inside Blender's scripting tab.
Requires: kinematics_{n}rotors.csv in the same directory as this script.

Creates an Empty object and inserts location + rotation keyframes
for every recorded timestep.  Set your scene FPS to match the export.
"""
import bpy, csv, math
from pathlib import Path

CSV = Path(__file__).parent / "kinematics_{n}rotors.csv"
FPS = round(1.0 / {result["dt"]:.4f})   # match simulation dt

# Remove existing "Drone" object if present
if "Drone" in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects["Drone"], do_unlink=True)

bpy.ops.object.empty_add(type="ARROWS")
obj = bpy.context.active_object
obj.name = "Drone"
obj.rotation_mode = "QUATERNION"
bpy.context.scene.render.fps = FPS

with open(CSV, newline="") as f:
    reader = csv.reader(row for row in f if not row.startswith("#"))
    header = next(reader)
    for row in reader:
        d = dict(zip(header, row))
        frame = round(float(d["time_s"]) * FPS) + 1
        bpy.context.scene.frame_set(frame)
        obj.location = (float(d["x_m"]), float(d["y_m"]), float(d["z_m"]))
        obj.rotation_quaternion = (
            float(d["qw"]), float(d["qx"]), float(d["qy"]), float(d["qz"])
        )
        obj.keyframe_insert("location", frame=frame)
        obj.keyframe_insert("rotation_quaternion", frame=frame)

print(f"Imported {{frame}} frames into Blender object \\'Drone\\'.")
'''
    bl_path = out_dir / f"blender_import_{n}rotors.py"
    bl_path.write_text(blender_script, encoding="utf-8")
    print(f"  Saved {bl_path.name}  (run inside Blender's Script editor)")


# =============================================================================
# Tracking time-series plot
# =============================================================================

def plot_tracking(result: dict, path: InspectionPath) -> None:
    """Four-panel tracking quality figure saved to results/."""
    n   = result["n_rotors"]
    t   = result["t"]
    pos = result["pos"]
    tgt = result["target"]
    err = result["error"]
    bat = result["battery"]
    pwr = result["power"]
    att = result["attitude"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Path-Following — {n}-Rotor Drone\n"
        f"Mean tracking error: {err.mean():.2f} m  |  "
        f"Peak error: {err.max():.2f} m  |  "
        f"Battery remaining: {bat[-1]/result['bat_cap']*100:.1f} %",
        fontsize=11, fontweight="bold", y=0.98)

    gs   = gridspec.GridSpec(4, 1, hspace=0.45,
                             top=0.93, bottom=0.07, left=0.09, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    col  = ROTOR_COLORS.get(n, "#333333")

    # ── Panel 0: XY top-down trajectory ──────────────────────────────────────
    ax = axes[0]
    ax.plot(path.xyz[:, 0], path.xyz[:, 1],
            "k--", lw=0.8, alpha=0.4, label="Path")
    ax.plot(tgt[:, 0], tgt[:, 1],
            "r-", lw=1.0, alpha=0.6, label="Target")
    ax.plot(pos[:, 0], pos[:, 1],
            color=col, lw=1.2, label="Drone")
    ax.plot(pos[0, 0],  pos[0, 1],  "go", ms=6, label="Start")
    ax.plot(pos[-1, 0], pos[-1, 1], "rs", ms=6, label="End")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Top-down (XY) trajectory", fontsize=9)
    ax.legend(fontsize=7, loc="best", ncol=3)
    ax.set_aspect("equal"); ax.grid(True, ls="--", lw=0.4, alpha=0.6)

    # ── Panel 1: Altitude profile ─────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, tgt[:, 2], "r--", lw=0.8, alpha=0.6, label="Target z")
    ax.plot(t, pos[:, 2], color=col, lw=1.2, label="Drone z")
    ax.set_ylabel("Altitude [m]"); ax.set_title("Altitude", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)

    # ── Panel 2: Tracking error ────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, err, color=col, lw=1.0, label="3-D error")
    ax.axhline(err.mean(), color="gray", ls="--", lw=0.8,
               label=f"Mean {err.mean():.2f} m")
    ax.fill_between(t, 0, err, alpha=0.25, color=col)
    ax.set_ylabel("Error [m]"); ax.set_title("3-D Position Tracking Error", fontsize=9)
    ax.legend(fontsize=7); ax.set_ylim(bottom=0)
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)

    # ── Panel 3: Power + battery ───────────────────────────────────────────────
    ax   = axes[3]
    ax.plot(t, pwr, color=col, lw=0.8, alpha=0.8, label="Motor power [W]")
    ax.set_ylabel("Power [W]"); ax.set_xlabel("Time [s]")
    ax.set_title("Motor Power & Battery", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(t, bat / result["bat_cap"] * 100,
             color="#1B5E20", lw=1.2, ls="--", label="Battery [%]")
    ax2.set_ylabel("Battery SoC [%]", color="#1B5E20")
    ax2.set_ylim(0, 110); ax2.tick_params(axis="y", labelcolor="#1B5E20")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)

    for ax in axes:
        if ax != axes[0]:
            ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelsize=8)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = RESULTS_DIR / f"tracking_{n}rotors.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname.name}")


# =============================================================================
# 3-D Animation
# =============================================================================

def animate(result: dict, path: InspectionPath,
            save: bool = False, speed: float = 1.0) -> None:
    """
    Animate the drone following the path in 3-D.

    Parameters
    ----------
    result  : output of run_simulation()
    path    : InspectionPath for the static trajectory line
    save    : if True write MP4 instead of showing interactively
    speed   : playback speed multiplier (2 = twice as fast)
    """
    n   = result["n_rotors"]
    pos = result["pos"]
    tgt = result["target"]
    att = result["attitude"]
    t   = result["t"]
    dt  = result["dt"]
    col = ROTOR_COLORS.get(n, "#333333")

    # Subsample for animation smoothness (target ~30 fps regardless of sim dt)
    stride = max(1, int(round(1.0 / (30.0 * dt * speed))))
    idx    = np.arange(0, len(t), stride)

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor("#0d1117")

    # Left: 3-D view
    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.set_facecolor("#0d1117")
    for p in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        p.pane.fill = False
        p.pane.set_edgecolor("#333333")

    # Right: live metrics (2 stacked panels)
    gs_r = gridspec.GridSpec(3, 1, left=0.55, right=0.97,
                              top=0.92, bottom=0.08, hspace=0.45)
    ax_err = fig.add_subplot(gs_r[0])
    ax_pow = fig.add_subplot(gs_r[1])
    ax_bat = fig.add_subplot(gs_r[2])
    for ax in [ax_err, ax_pow, ax_bat]:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.spines[:].set_color("#444444")
        ax.grid(True, ls="--", lw=0.3, alpha=0.4, color="#444444")

    # ── Static path line ─────────────────────────────────────────────────────
    xyz = path.xyz
    ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
              color="#666666", lw=0.8, alpha=0.5, label="Path")

    # ── Drone body axes (unit vectors, scaled) ────────────────────────────────
    AXIS_LEN = float(result["drone"].arm_length) * 1.5
    ax_colors = ["#FF6B6B", "#6BFF6B", "#6B6BFF"]   # x=red, y=green, z=blue
    axes_art  = [ax3d.quiver(0, 0, 0, 0, 0, 0, color=c, lw=1.5, alpha=0.9)
                 for c in ax_colors]

    # ── Rotor position markers ────────────────────────────────────────────────
    rotor_xy = result["drone"].rotor_xy   # (n, 2) in body frame
    n_r = n
    rpt, = ax3d.plot([], [], [], "o", color=col, ms=4, alpha=0.7)

    # ── Moving markers ────────────────────────────────────────────────────────
    tgt_pt,  = ax3d.plot([], [], [], "o", color="red",  ms=8,
                          alpha=0.9, label="Target")
    drone_pt, = ax3d.plot([], [], [], "D", color=col, ms=8,
                           alpha=0.9, label="Drone")
    trail_l,  = ax3d.plot([], [], [], "-", color=col, lw=0.8, alpha=0.35)
    TRAIL_LEN = 80 // max(stride, 1)

    # ── Metric plots (partial data at start) ─────────────────────────────────
    err_line, = ax_err.plot([], [], color="#FF6B6B", lw=1.0)
    pow_line, = ax_pow.plot([], [], color="#FFB86C", lw=1.0)
    bat_line, = ax_bat.plot([], [], color="#50FA7B", lw=1.0)

    ax_err.set_ylabel("Error [m]", fontsize=7)
    ax_err.set_title("Tracking error", fontsize=8)
    ax_pow.set_ylabel("Power [W]", fontsize=7)
    ax_pow.set_title("Motor power", fontsize=8)
    ax_bat.set_ylabel("Battery [%]", fontsize=7)
    ax_bat.set_title("Battery SoC", fontsize=8)
    ax_bat.set_xlabel("Time [s]", fontsize=7)
    ax_bat.set_ylim(0, 105)

    # Pre-compute axis limits
    all_xyz = np.vstack([pos, tgt, xyz])
    x_lim = (all_xyz[:, 0].min() - 5, all_xyz[:, 0].max() + 5)
    y_lim = (all_xyz[:, 1].min() - 5, all_xyz[:, 1].max() + 5)
    z_lim = (max(all_xyz[:, 2].min() - 3, 0), all_xyz[:, 2].max() + 5)
    ax3d.set_xlim(*x_lim); ax3d.set_ylim(*y_lim); ax3d.set_zlim(*z_lim)
    ax3d.set_xlabel("x [m]", color="white", fontsize=7)
    ax3d.set_ylabel("y [m]", color="white", fontsize=7)
    ax3d.set_zlabel("z [m]", color="white", fontsize=7)
    ax3d.tick_params(colors="white", labelsize=6)

    ax_err.set_xlim(t[0], t[-1])
    ax_err.set_ylim(0, result["error"].max() * 1.15 + 0.5)
    ax_pow.set_xlim(t[0], t[-1])
    ax_pow.set_ylim(0, result["power"].max() * 1.15 + 100)
    ax_bat.set_xlim(t[0], t[-1])

    title_txt = fig.suptitle("", color="white", fontsize=10, fontweight="bold")

    from drone import rotation_matrix  # noqa — already imported indirectly

    def update(frame_idx):
        i = idx[frame_idx]
        p   = pos[i]
        tg  = tgt[i]
        phi, theta, psi = att[i]
        R   = rotation_matrix(phi, theta, psi)

        # Target and drone markers
        tgt_pt.set_data([tg[0]], [tg[1]])
        tgt_pt.set_3d_properties([tg[2]])
        drone_pt.set_data([p[0]], [p[1]])
        drone_pt.set_3d_properties([p[2]])

        # Trail
        lo = max(0, i - TRAIL_LEN * stride)
        trail_l.set_data(pos[lo:i:stride, 0], pos[lo:i:stride, 1])
        trail_l.set_3d_properties(pos[lo:i:stride, 2])

        # Body-frame axes
        for k, axis_art in enumerate(axes_art):
            axis_art.remove()
            d = R[:, k] * AXIS_LEN
            axes_art[k] = ax3d.quiver(
                p[0], p[1], p[2],
                d[0], d[1], d[2],
                color=ax_colors[k], lw=1.5, alpha=0.85
            )

        # Rotor dots (body xy → world positions)
        r_world = p[None, :] + (R[:, :2] @ rotor_xy.T).T  # (n, 3)
        rpt.set_data(r_world[:, 0], r_world[:, 1])
        rpt.set_3d_properties(r_world[:, 2])

        # Metric traces up to current frame
        si = idx[:frame_idx + 1]
        err_line.set_data(t[si], result["error"][si])
        pow_line.set_data(t[si], result["power"][si])
        bat_line.set_data(t[si], result["battery"][si] / result["bat_cap"] * 100)

        title_txt.set_text(
            f"{n}-rotor drone path following  |  "
            f"t = {t[i]:.1f} s  |  "
            f"error = {result['error'][i]:.2f} m  |  "
            f"battery = {result['battery'][i]/result['bat_cap']*100:.1f} %"
        )
        return ([tgt_pt, drone_pt, trail_l, rpt, err_line, pow_line, bat_line]
                + axes_art)

    interval_ms = int(dt * stride * 1000 / speed)

    handles = [
        mpatches.Patch(color="#666666", alpha=0.5, label="Path"),
        mpatches.Patch(color="red",     alpha=0.8, label="Target"),
        mpatches.Patch(color=col,       alpha=0.8, label="Drone"),
    ]
    ax3d.legend(handles=handles, loc="upper left", fontsize=7,
                facecolor="#161b22", labelcolor="white")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if save:
        # ── Video export: use FuncAnimation ──────────────────────────────────
        anim = FuncAnimation(fig, update, frames=len(idx),
                             interval=interval_ms, blit=False)
        fname = RESULTS_DIR / f"animation_{n}rotors.mp4"
        anim.save(str(fname), fps=30, dpi=120,
                  writer="ffmpeg",
                  extra_args=["-preset", "fast", "-crf", "23"])
        print(f"  Saved {fname.name}")
        plt.close(fig)

    else:
        # ── Interactive: manual timer so the slider can seek freely ──────────
        fig.subplots_adjust(bottom=0.14)

        # Time slider ─────────────────────────────────────────────────────────
        def _dark_ax(rect):
            ax = fig.add_axes(rect)
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#888888")
            for sp in ax.spines.values():
                sp.set_color("#444444")
            return ax

        sl_ax = _dark_ax([0.12, 0.062, 0.52, 0.025])
        time_slider = Slider(sl_ax, "t (s)", t[idx[0]], t[idx[-1]],
                             valinit=t[idx[0]], color="#2196F3",
                             track_color="#21262d")
        time_slider.label.set_color("white")
        time_slider.label.set_fontsize(8)
        time_slider.valtext.set_color("#aaaaaa")
        time_slider.valtext.set_fontsize(7)

        # Buttons ─────────────────────────────────────────────────────────────
        def _btn(rect, label):
            ax = _dark_ax(rect)
            b  = Button(ax, label, color="#21262d", hovercolor="#30363d")
            b.label.set_color("white")
            b.label.set_fontsize(9)
            return b

        rewind_btn = _btn([0.67, 0.025, 0.10, 0.045], "|<  Start")
        play_btn   = _btn([0.79, 0.025, 0.10, 0.045], "||  Pause")

        # Shared mutable state ────────────────────────────────────────────────
        state = {"i": 0, "playing": True, "slider_driven": False}

        # Draw one frame without blitting
        def draw(fi: int):
            update(fi)
            # Sync slider without re-triggering its callback
            state["slider_driven"] = True
            time_slider.set_val(t[idx[fi]])
            state["slider_driven"] = False
            fig.canvas.draw_idle()

        draw(0)

        # Timer callback (auto-advance while playing) ─────────────────────────
        def on_timer():
            if not state["playing"]:
                return
            fi = state["i"] + 1
            if fi >= len(idx):
                state["playing"] = False
                play_btn.label.set_text("▶  Play")
                fig.canvas.draw_idle()
                return
            state["i"] = fi
            draw(fi)

        timer = fig.canvas.new_timer(interval=interval_ms)
        timer.add_callback(on_timer)
        timer.start()

        # Slider callback (scrub to any time) ─────────────────────────────────
        def on_slider(val):
            if state["slider_driven"]:
                return
            fi = int(np.argmin(np.abs(t[idx] - val)))
            state["i"] = fi
            update(fi)
            fig.canvas.draw_idle()

        time_slider.on_changed(on_slider)

        # Play / Pause ────────────────────────────────────────────────────────
        def toggle_pause(event=None):
            state["playing"] = not state["playing"]
            play_btn.label.set_text(
                "⏸  Pause" if state["playing"] else "▶  Play"
            )
            fig.canvas.draw_idle()

        play_btn.on_clicked(toggle_pause)

        # Rewind to start ─────────────────────────────────────────────────────
        def rewind(event=None):
            state["i"] = 0
            draw(0)

        rewind_btn.on_clicked(rewind)

        # Keyboard shortcuts ──────────────────────────────────────────────────
        def on_key(event):
            if event.key == " ":
                toggle_pause()
            elif event.key == "r":
                rewind()
            elif event.key == "left":       # ← step back 5 %
                fi = max(state["i"] - max(1, len(idx) // 20), 0)
                state["i"] = fi
                draw(fi)
            elif event.key == "right":      # → step forward 5 %
                fi = min(state["i"] + max(1, len(idx) // 20), len(idx) - 1)
                state["i"] = fi
                draw(fi)

        fig.canvas.mpl_connect("key_press_event", on_key)

        plt.show()
        timer.stop()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Drone path-following simulation")
    parser.add_argument("--path",   default=None,
                        help="Path CSV file (default: generate sample helix)")
    parser.add_argument("--rotors", type=int, default=None,
                        help="Number of rotors: 4, 6, or 8 (default: from config)")
    parser.add_argument("--save",   action="store_true",
                        help="Save animation to MP4 instead of showing")
    parser.add_argument("--no-anim", action="store_true",
                        help="Skip animation (only produce tracking plot)")
    parser.add_argument("--config", default=str(CONFIG_PATH),
                        help="Path to config YAML")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config))
    sim_cfg = config.get("simulation", {})

    # ── Path ─────────────────────────────────────────────────────────────────
    if args.path is None:
        sample = str(_HERE / "sample_helix.csv")
        if not Path(sample).exists():
            data = helix_path(radius=20.0, z_start=10.0, z_end=35.0,
                              n_turns=2.0, duration=60.0, dt=0.1)
            save_csv(data, sample)
        path_file = sample
    else:
        path_file = args.path

    inspection_path = InspectionPath(path_file)
    print(f"  Path loaded: {path_file}")
    print(f"  Duration: {inspection_path.total_time:.1f} s  |  "
          f"Points: {len(inspection_path.times)}")

    # ── Rotor count ───────────────────────────────────────────────────────────
    n_rotors = args.rotors or int(config["num_motors"])
    if n_rotors not in (4, 6, 8):
        print(f"  ERROR: --rotors must be 4, 6, or 8 (got {n_rotors})")
        sys.exit(1)

    print(f"\n  Simulating {n_rotors}-rotor configuration …", flush=True)

    # ── Simulation ────────────────────────────────────────────────────────────
    result = run_simulation(config, inspection_path, n_rotors)

    t_sim  = result["t"][-1]
    t_plan = inspection_path.total_time
    completed = t_sim >= t_plan - result["dt"] * 2

    print(f"  done — "
          f"{'COMPLETE' if completed else f'ended at {t_sim:.1f}/{t_plan:.1f} s (battery?)'}")
    print(f"  Mean error: {result['error'].mean():.3f} m  "
          f"Peak: {result['error'].max():.3f} m")
    print(f"  Battery remaining: "
          f"{result['battery'][-1]/result['bat_cap']*100:.1f} %."
          f"  Total energy used: {(result['bat_cap'] - result['battery'][-1]):.2f} Wh"
          f"  Total battery left: {result['battery'][-1]:.2f} Wh")

    # ── Kinematics export ─────────────────────────────────────────────────────
    print(f"\n  Exporting kinematics …")
    export_kinematics(result, RESULTS_DIR)

    # ── Tracking plot ─────────────────────────────────────────────────────────
    print(f"\n  Generating tracking plot …")
    plot_tracking(result, inspection_path)

    # ── Animation ─────────────────────────────────────────────────────────────
    if not args.no_anim:
        save_anim = args.save or bool(sim_cfg.get("save_animation", False))
        speed     = float(sim_cfg.get("animation_speed", 1.0))
        print(f"\n  {'Saving' if save_anim else 'Showing'} animation "
              f"(speed ×{speed}) …")
        if save_anim:
            matplotlib.use("Agg")
        animate(result, inspection_path, save=save_anim, speed=speed)

    print("\n  Done.  Results in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
