#!/usr/bin/env python3
"""
Wind turbine tower – 2D LiDAR inspection simulation.

Two windows are shown simultaneously:
  Window 1 – Overview  : world-frame map with drone trail, LiDAR returns,
                         heading arrows, algorithm info panel and error chart.
  Window 2 – LiDAR Feed: raw point cloud only, in the drone's own body frame
                         (drone at origin, facing +X), with range rings and
                         the fitted circle overlaid.  This is what the sensor
                         itself "sees", free of any world context.

At the end of the run every scan is written to  lidar_data.csv  (one LiDAR
point per row:  t, drone_x, drone_y, heading_deg, pt_x_world, pt_y_world,
pt_x_body, pt_y_body) so the data can be loaded in any tool.

Run with:
    conda activate aquaerial && python lidar_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation
from collections import deque
import csv
import pathlib

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CYLINDER_POS    = np.array([0.0, 0.0])
CYLINDER_RADIUS = 2.5            # metres  (~5 m diameter, typical turbine base)

LIDAR_RESOLUTION = 1.0           # degrees per ray  (360 rays per scan)
LIDAR_MAX_RANGE  = 25.0          # metres
LIDAR_NOISE_STD  = 0.05          # metres (Gaussian range noise)

ORBIT_RADIUS_MEAN = 8.0          # mean orbit radius (m)
ORBIT_RADIUS_AMP  = 1.5          # sinusoidal variation amplitude (m)
ORBIT_SPEED       = 0.28         # rad s⁻¹

HEADING_GAIN  = 1.5              # P-controller gain  (rad/s per rad of error)

SIM_DURATION  = 25.0             # seconds
DT            = 0.05             # simulation / animation time step (s)
ANIM_INTERVAL = 50               # ms between animation frames
TRAIL_LEN     = 400

ARROW_LEN  = 2.5                 # heading arrow length in the overview map (m)
WORLD_HALF = 14                  # half-extent of the overview map (m)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (GitHub dark theme inspired)
# ─────────────────────────────────────────────────────────────────────────────

C = dict(
    bg     = '#0d1117',
    panel  = '#161b22',
    grid   = '#21262d',
    border = '#30363d',
    dim    = '#8b949e',
    text   = '#f0f6fc',
    green  = '#3fb950',
    orange = '#ffa657',
    red    = '#ff7b72',
    blue   = '#58a6ff',
    purple = '#d2a8ff',
)


# ─────────────────────────────────────────────────────────────────────────────
# LiDAR ray-casting
# ─────────────────────────────────────────────────────────────────────────────

def ray_circle_hit(origin, direction, centre, radius, max_range):
    """Closest positive range to a circle intersection, or None."""
    d = origin - centre
    b = 2.0 * np.dot(d, direction)
    c = np.dot(d, d) - radius ** 2
    disc = b * b - 4.0 * c          # a = 1 (unit direction)
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    for t in ((-b - sq) / 2.0, (-b + sq) / 2.0):
        if 1e-3 < t <= max_range:
            return t
    return None


def lidar_scan(pos, cyl_pos, cyl_r):
    """
    Full 360° LiDAR scan from `pos`.
    Returns (N, 2) world-frame hit points with Gaussian range noise.
    """
    angles = np.deg2rad(np.arange(0.0, 360.0, LIDAR_RESOLUTION))
    hits = []
    for a in angles:
        direction = np.array([np.cos(a), np.sin(a)])
        r = ray_circle_hit(pos, direction, cyl_pos, cyl_r, LIDAR_MAX_RANGE)
        if r is not None:
            noisy_r = max(0.01, r + np.random.normal(0.0, LIDAR_NOISE_STD))
            hits.append(pos + noisy_r * direction)
    return np.array(hits) if hits else np.empty((0, 2))


def to_body_frame(world_pts, drone_pos, drone_heading):
    """
    Rigid-body transform: world → drone body frame.
    Drone is placed at the origin, its heading direction becomes +X.

    Works for both (N, 2) arrays and a single (2,) point.
    """
    pts = np.atleast_2d(world_pts)
    if len(pts) == 0:
        return np.empty((0, 2))
    translated = pts - drone_pos
    c, s = np.cos(-drone_heading), np.sin(-drone_heading)
    R = np.array([[c, -s], [s, c]])
    result = (R @ translated.T).T
    return result[0] if np.ndim(world_pts) == 1 else result


# ─────────────────────────────────────────────────────────────────────────────
# Circle fitting – algebraic least-squares
# ─────────────────────────────────────────────────────────────────────────────

def fit_circle(pts):
    """
    Fit  x² + y² + Ax + By + C = 0  to 2-D points.
    Returns (centre_xy, radius) or (None, None).
    """
    if len(pts) < 5:
        return None, None
    x, y = pts[:, 0], pts[:, 1]
    A_mat = np.column_stack([x, y, np.ones(len(x))])
    b_vec = -(x ** 2 + y ** 2)
    coef, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    cx, cy = -coef[0] / 2.0, -coef[1] / 2.0
    r2 = cx ** 2 + cy ** 2 - coef[2]
    if r2 <= 0:
        return None, None
    return np.array([cx, cy]), float(np.sqrt(r2))


# ─────────────────────────────────────────────────────────────────────────────
# Orientation algorithm
# ─────────────────────────────────────────────────────────────────────────────

def orient_toward_centre(pos, heading, pts):
    """
    Fit a circle to LiDAR returns; compute yaw correction to face the centre.
    Returns (est_centre, desired_heading, angular_error) or all None.
    angular_error is wrapped to (−π, π].
    """
    centre, _ = fit_circle(pts)
    if centre is None:
        return None, None, None
    delta   = centre - pos
    desired = np.arctan2(delta[1], delta[0])
    error   = desired - heading
    error   = (error + np.pi) % (2.0 * np.pi) - np.pi
    return centre, desired, error


# ─────────────────────────────────────────────────────────────────────────────
# Drone state machine
# ─────────────────────────────────────────────────────────────────────────────

class Drone:
    """
    Position follows a kinematic orbit; heading is corrected by a P-controller
    driven by the orientation algorithm.  Starts with a tangential heading
    (~90° off from facing the centre) to give the controller a visible transient.
    """

    def __init__(self):
        theta0       = np.pi
        self.pos     = CYLINDER_POS + ORBIT_RADIUS_MEAN * np.array([np.cos(theta0), np.sin(theta0)])
        self.heading = theta0 + np.pi / 2.0   # tangential → 90° off from centre
        self.t       = 0.0
        self.trail   = deque(maxlen=TRAIL_LEN)

    def step(self):
        self.t  += DT
        theta    = ORBIT_SPEED * self.t + np.pi
        r        = ORBIT_RADIUS_MEAN + ORBIT_RADIUS_AMP * np.sin(0.15 * self.t + 0.5)
        self.pos = CYLINDER_POS + r * np.array([np.cos(theta), np.sin(theta)])
        self.trail.append(self.pos.copy())

        pts                       = lidar_scan(self.pos, CYLINDER_POS, CYLINDER_RADIUS)
        est_ctr, des_hdg, ang_err = orient_toward_centre(self.pos, self.heading, pts)

        if ang_err is not None:
            self.heading += HEADING_GAIN * ang_err * DT

        return pts, est_ctr, des_hdg, ang_err


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(C['panel'])
    ax.tick_params(colors=C['dim'])
    for sp in ax.spines.values():
        sp.set_edgecolor(C['border'])


def _make_arrow(ax, color, **kw):
    akw = dict(arrowstyle='->', linewidth=2.5, mutation_scale=18, zorder=9)
    akw.update(kw)
    p = FancyArrowPatch((0, 0), (1, 0), color=color, **akw)
    ax.add_patch(p)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Window 1 – Overview
# ─────────────────────────────────────────────────────────────────────────────

def build_overview():
    fig = plt.figure(figsize=(15, 7), facecolor=C['bg'])
    fig.canvas.manager.set_window_title('LiDAR Inspection – Overview')
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.4, 1],
                           height_ratios=[1.5, 1], wspace=0.38, hspace=0.5)
    ax_map  = fig.add_subplot(gs[:, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_err  = fig.add_subplot(gs[1, 1])

    for ax in (ax_map, ax_info, ax_err):
        _style_ax(ax)

    # ── map ──────────────────────────────────────────────────────────────────
    ax_map.set(xlim=(-WORLD_HALF, WORLD_HALF), ylim=(-WORLD_HALF, WORLD_HALF), aspect='equal')
    ax_map.set_title('2D LiDAR Scan  –  World Frame (Top View)', color=C['text'], fontsize=12, pad=8)
    ax_map.set_xlabel('X (m)', color=C['dim'])
    ax_map.set_ylabel('Y (m)', color=C['dim'])
    ax_map.grid(True, color=C['grid'], linewidth=0.5, zorder=0)

    ax_map.add_patch(plt.Circle(CYLINDER_POS, CYLINDER_RADIUS,
                                facecolor='#1c3a5e', edgecolor=C['blue'], lw=2, zorder=2))
    ax_map.plot(*CYLINDER_POS, '+', color=C['blue'], ms=12, mew=2, zorder=3)
    ax_map.text(0.0, -CYLINDER_RADIUS - 0.8, 'Tower axis',
                color=C['blue'], fontsize=8, ha='center')

    trail_ln, = ax_map.plot([], [], '-', color='#484f58', lw=1, alpha=0.6, zorder=4)
    lidar_sc   = ax_map.scatter([], [], s=5, c=C['green'], alpha=0.85, zorder=6)
    drone_ln, = ax_map.plot([], [], 'o', color=C['text'], ms=9, zorder=10)
    est_ln,   = ax_map.plot([], [], '*', color=C['red'],  ms=14, zorder=11)
    cur_arrow = _make_arrow(ax_map, C['orange'])
    des_arrow = _make_arrow(ax_map, C['red'])

    ax_map.legend(handles=[
        mpatches.Patch(color=C['green'],  label='LiDAR returns'),
        mpatches.Patch(color=C['orange'], label='Current heading'),
        mpatches.Patch(color=C['red'],    label='Desired heading (to centre)'),
        mpatches.Patch(color=C['blue'],   label='True cylinder'),
    ], loc='upper right', facecolor=C['panel'], labelcolor=C['text'],
       edgecolor=C['border'], fontsize=8)

    # ── info panel ───────────────────────────────────────────────────────────
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1); ax_info.axis('off')
    ax_info.set_title('Algorithm Output', color=C['text'], fontsize=11, pad=6)

    def _row(y, label):
        ax_info.text(0.04, y, label, color=C['dim'], fontsize=9, va='top',
                     transform=ax_info.transAxes, fontfamily='monospace')
        return ax_info.text(0.04, y - 0.09, '—', color=C['text'], fontsize=10,
                            va='top', transform=ax_info.transAxes,
                            fontfamily='monospace', fontweight='bold')

    rows = [_row(y, lbl) for y, lbl in [
        (0.93, 'Time'), (0.80, 'Drone pos'), (0.67, 'Est. centre'),
        (0.54, 'Centre error'), (0.41, 'Angular error'), (0.28, 'LiDAR points'),
        (0.15, 'Status'),
    ]]

    ax_info.add_patch(mpatches.Rectangle((0.04, 0.04), 0.92, 0.05,
        facecolor=C['grid'], edgecolor=C['border'], lw=1, transform=ax_info.transAxes))
    gauge = mpatches.Rectangle((0.5, 0.04), 0.0, 0.05,
        facecolor=C['green'], transform=ax_info.transAxes)
    ax_info.add_patch(gauge)
    for txt, xa, ha in [('−180°', 0.04, 'left'), ('0°', 0.5, 'center'), ('+180°', 0.96, 'right')]:
        ax_info.text(xa, 0.02, txt, color=C['dim'], fontsize=7,
                     transform=ax_info.transAxes, ha=ha, va='top')

    # ── error chart ──────────────────────────────────────────────────────────
    ax_err.set_title('Angular Error over Time', color=C['text'], fontsize=11, pad=6)
    ax_err.set_xlabel('Time (s)', color=C['dim'], fontsize=9)
    ax_err.set_ylabel('|Error| (°)', color=C['dim'], fontsize=9)
    ax_err.set_xlim(0, SIM_DURATION); ax_err.set_ylim(0, 100)
    ax_err.grid(True, color=C['grid'], linewidth=0.5)
    ax_err.axhline(10, color=C['green'], lw=1, ls='--', alpha=0.7)
    ax_err.text(0.5, 13, 'Aligned  (< 10°)', color=C['green'],
                fontsize=7, transform=ax_err.transAxes, ha='center')
    err_line, = ax_err.plot([], [], '-', color=C['orange'], lw=1.5)

    fig.suptitle('Wind Turbine Tower  –  LiDAR Orientation Demo',
                 color=C['text'], fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    elems = dict(trail_ln=trail_ln, lidar_sc=lidar_sc, drone_ln=drone_ln,
                 est_ln=est_ln, cur_arrow=cur_arrow, des_arrow=des_arrow,
                 rows=rows, gauge=gauge, err_line=err_line)
    return fig, elems


# ─────────────────────────────────────────────────────────────────────────────
# Window 2 – Raw LiDAR Feed (drone body frame)
# ─────────────────────────────────────────────────────────────────────────────

def build_lidar_window():
    """
    A dedicated window showing only raw LiDAR returns, in the drone's body
    frame (drone at origin, heading along +X).  Range rings and the fitted
    circle give spatial context without leaking world-frame information.
    """
    fig2, ax = plt.subplots(1, 1, figsize=(7, 7), facecolor=C['bg'])
    fig2.canvas.manager.set_window_title('LiDAR Inspection – Raw LiDAR Feed')
    _style_ax(ax)

    VIEW = 14
    ax.set(xlim=(-VIEW, VIEW), ylim=(-VIEW, VIEW), aspect='equal')
    ax.set_title('Raw LiDAR Point Cloud  –  Drone Body Frame\n'
                 '(Drone at origin, heading along +X)',
                 color=C['text'], fontsize=11, pad=8)
    ax.set_xlabel('Body-X  /  Forward (m)', color=C['dim'])
    ax.set_ylabel('Body-Y  /  Left (m)',     color=C['dim'])
    ax.grid(True, color=C['grid'], linewidth=0.5, zorder=0)

    # Range rings
    for r_ring in (3, 5, 8, 10, 13):
        ax.add_patch(plt.Circle((0, 0), r_ring, fill=False,
                                edgecolor=C['grid'], lw=0.8, ls='--', zorder=1))
        ax.text(r_ring + 0.2, 0.4, f'{r_ring} m',
                color=C['dim'], fontsize=7, va='bottom')

    # Drone symbol (static – always at origin in body frame)
    ax.plot(0, 0, 'o', color=C['orange'], ms=8, zorder=12, markeredgewidth=0)
    ax.plot(0, 0, '+', color=C['orange'], ms=14, mew=2, zorder=12)

    # Static heading arrow along +X
    hdg_arrow_static = _make_arrow(ax, C['orange'], linewidth=2,
                                   mutation_scale=15, zorder=11)
    hdg_arrow_static.set_positions((0, 0), (ARROW_LEN, 0))
    ax.text(ARROW_LEN + 0.3, 0.15, 'heading', color=C['orange'], fontsize=7)

    # Dynamic elements
    raw_sc     = ax.scatter([], [], s=7, c=C['green'], alpha=0.9, zorder=6)
    fit_circle_patch = plt.Circle((0, 0), 1, fill=False,
                                  edgecolor=C['red'], lw=1.5, zorder=7)
    fit_centre_ln, = ax.plot([], [], '*', color=C['red'], ms=13, zorder=8)
    ctr_arrow  = _make_arrow(ax, C['red'], linewidth=2, mutation_scale=15, zorder=10)
    ax.add_patch(fit_circle_patch)

    # Text overlays (top-left corner)
    ov_time = ax.text(0.03, 0.97, '', color=C['text'],   fontsize=9, va='top',
                      transform=ax.transAxes, fontfamily='monospace')
    ov_pts  = ax.text(0.03, 0.92, '', color=C['green'],  fontsize=9, va='top',
                      transform=ax.transAxes, fontfamily='monospace')
    ov_rng  = ax.text(0.03, 0.87, '', color=C['dim'],    fontsize=9, va='top',
                      transform=ax.transAxes, fontfamily='monospace')
    ov_ctr  = ax.text(0.03, 0.82, '', color=C['red'],    fontsize=9, va='top',
                      transform=ax.transAxes, fontfamily='monospace')
    ov_err  = ax.text(0.03, 0.77, '', color=C['orange'], fontsize=9, va='top',
                      transform=ax.transAxes, fontfamily='monospace')

    ax.legend(handles=[
        mpatches.Patch(color=C['green'],  label='LiDAR returns'),
        mpatches.Patch(color=C['orange'], label='Drone heading (+X)'),
        mpatches.Patch(color=C['red'],    label='Fitted circle / centre'),
    ], loc='upper right', facecolor=C['panel'], labelcolor=C['text'],
       edgecolor=C['border'], fontsize=8)

    plt.tight_layout()

    elems = dict(ax=ax, raw_sc=raw_sc,
                 fit_circle_patch=fit_circle_patch,
                 fit_centre_ln=fit_centre_ln,
                 ctr_arrow=ctr_arrow,
                 ov_time=ov_time, ov_pts=ov_pts,
                 ov_rng=ov_rng, ov_ctr=ov_ctr, ov_err=ov_err)
    return fig2, elems


# ─────────────────────────────────────────────────────────────────────────────
# Data logger
# ─────────────────────────────────────────────────────────────────────────────

class ScanLogger:
    """Collects per-point rows; writes CSV on flush()."""

    HEADER = ('t_s', 'drone_x_m', 'drone_y_m', 'heading_deg',
              'pt_x_world_m', 'pt_y_world_m', 'pt_x_body_m', 'pt_y_body_m')

    def __init__(self):
        self._rows = []

    def record(self, t, drone_pos, drone_heading, world_pts, body_pts):
        hdg_deg = float(np.rad2deg(drone_heading))
        for (wx, wy), (bx, by) in zip(world_pts, body_pts):
            self._rows.append((round(t, 4),
                                round(float(drone_pos[0]), 4),
                                round(float(drone_pos[1]), 4),
                                round(hdg_deg, 4),
                                round(float(wx), 4), round(float(wy), 4),
                                round(float(bx), 4), round(float(by), 4)))

    def flush(self, path='lidar_data.csv'):
        out = pathlib.Path(path)
        with out.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(self.HEADER)
            w.writerows(self._rows)
        print(f'[export]  {len(self._rows):,} LiDAR points saved → {out.resolve()}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run():
    np.random.seed(42)
    drone  = Drone()
    logger = ScanLogger()

    fig1, e1 = build_overview()
    fig2, e2 = build_lidar_window()

    times_hist: list = []
    errs_hist:  list = []
    total_frames = int(SIM_DURATION / DT)

    def update(frame):
        pts, est_ctr, des_hdg, ang_err = drone.step()
        t, pos, hdg = drone.t, drone.pos, drone.heading

        # Body-frame LiDAR points
        body_pts = to_body_frame(pts, pos, hdg) if len(pts) > 0 else np.empty((0, 2))

        # ── Log data ─────────────────────────────────────────────────────────
        if len(pts) > 0:
            logger.record(t, pos, hdg, pts, body_pts)

        # ── Window 1: Overview ────────────────────────────────────────────────
        if len(drone.trail) > 1:
            e1['trail_ln'].set_data([p[0] for p in drone.trail],
                                    [p[1] for p in drone.trail])
        e1['lidar_sc'].set_offsets(pts if len(pts) > 0 else np.empty((0, 2)))
        e1['drone_ln'].set_data([pos[0]], [pos[1]])

        cur_end = pos + ARROW_LEN * np.array([np.cos(hdg), np.sin(hdg)])
        e1['cur_arrow'].set_positions(tuple(pos), tuple(cur_end))

        if des_hdg is not None:
            des_end = pos + ARROW_LEN * np.array([np.cos(des_hdg), np.sin(des_hdg)])
            e1['des_arrow'].set_positions(tuple(pos), tuple(des_end))
            e1['des_arrow'].set_visible(True)
            e1['est_ln'].set_data([est_ctr[0]], [est_ctr[1]])
            e1['est_ln'].set_visible(True)
        else:
            e1['des_arrow'].set_visible(False)
            e1['est_ln'].set_visible(False)

        t_txt, pos_txt, ectr_txt, cerr_txt, aerr_txt, npts_txt, stat_txt = e1['rows']
        t_txt.set_text(f'{t:.1f} s')
        pos_txt.set_text(f'({pos[0]:+.2f},  {pos[1]:+.2f}) m')
        npts_txt.set_text(str(len(pts)))

        if est_ctr is not None:
            ectr_txt.set_text(f'({est_ctr[0]:+.2f},  {est_ctr[1]:+.2f}) m')
            cerr_txt.set_text(f'{np.linalg.norm(est_ctr - CYLINDER_POS):.3f} m')

        if ang_err is not None:
            deg = np.rad2deg(ang_err)
            col = C['green'] if abs(deg) < 10 else (C['orange'] if abs(deg) < 30 else C['red'])
            aerr_txt.set_text(f'{deg:+.1f}°')
            aerr_txt.set_color(col)
            stat_txt.set_text('ALIGNED' if abs(deg) < 10 else 'CORRECTING')
            stat_txt.set_color(C['green'] if abs(deg) < 10 else C['orange'])
            norm  = ang_err / np.pi
            bar_w = 0.46 * abs(norm)
            e1['gauge'].set_x(0.5 if norm >= 0 else 0.5 - bar_w)
            e1['gauge'].set_width(bar_w)
            e1['gauge'].set_facecolor(col)
            times_hist.append(t)
            errs_hist.append(abs(deg))
            e1['err_line'].set_data(times_hist, errs_hist)

        # ── Window 2: Raw LiDAR feed ──────────────────────────────────────────
        e2['raw_sc'].set_offsets(body_pts if len(body_pts) > 0 else np.empty((0, 2)))

        if est_ctr is not None:
            body_ctr = to_body_frame(est_ctr, pos, hdg)
            _, est_r  = fit_circle(pts)          # radius in world = body (rotation-invariant)

            # Fitted circle
            if est_r is not None:
                e2['fit_circle_patch'].set_center(tuple(body_ctr))
                e2['fit_circle_patch'].set_radius(est_r)
                e2['fit_circle_patch'].set_visible(True)

            # Centre star
            e2['fit_centre_ln'].set_data([body_ctr[0]], [body_ctr[1]])
            e2['fit_centre_ln'].set_visible(True)

            # Arrow from drone origin to estimated centre
            dist = np.linalg.norm(body_ctr)
            if dist > 0.1:
                tip = body_ctr * (1.0 - min(0.5 / dist, 0.3))   # stop just short
                e2['ctr_arrow'].set_positions((0, 0), tuple(tip))
                e2['ctr_arrow'].set_visible(True)
            else:
                e2['ctr_arrow'].set_visible(False)

            # Overlays
            e2['ov_ctr'].set_text(
                f'Est. ctr  ({body_ctr[0]:+.1f}, {body_ctr[1]:+.1f}) m  '
                f'| r={est_r:.2f} m' if est_r else '')
            if ang_err is not None:
                deg = np.rad2deg(ang_err)
                col = C['green'] if abs(deg) < 10 else (C['orange'] if abs(deg) < 30 else C['red'])
                e2['ov_err'].set_text(f'Angular error  {deg:+.1f}°')
                e2['ov_err'].set_color(col)
        else:
            e2['fit_circle_patch'].set_visible(False)
            e2['fit_centre_ln'].set_visible(False)
            e2['ctr_arrow'].set_visible(False)
            e2['ov_ctr'].set_text('')
            e2['ov_err'].set_text('')

        e2['ov_time'].set_text(f't = {t:.1f} s')
        e2['ov_pts'].set_text(f'{len(pts)} LiDAR returns')
        if len(body_pts) > 0:
            ranges = np.linalg.norm(body_pts, axis=1)
            e2['ov_rng'].set_text(f'Range  {ranges.min():.1f} – {ranges.max():.1f} m')

        fig2.canvas.draw_idle()

        # ── Export on last frame ──────────────────────────────────────────────
        if frame == total_frames - 1:
            logger.flush('lidar_data.csv')

    ani = FuncAnimation(fig1, update, frames=total_frames,
                        interval=ANIM_INTERVAL, blit=False, repeat=False)
    plt.show()
    return ani


if __name__ == '__main__':
    run()
