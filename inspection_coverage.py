# coverage.py  —  drone sensor coverage analysis
# Supports pill-shaped (aerial) and box-shaped (underwater) bodies.
# Sensors placed on body faces; each sensor projects a cone of coverage.
# Monte Carlo integration gives the % of the full sphere captured.
# Body frame: x forward, y left, z up, origin at body centre.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from aerial_props import D as _ap_D

# ──────────────────────────────────────────────────────────────────────────────
#  SHARED
# ──────────────────────────────────────────────────────────────────────────────
N_MC = 1_000_000   # [-]  Monte Carlo samples (more = more accurate, ~±0.1% at 1M)

# Detection-sphere radius: points sampled on a sphere of this radius around the
# drone centre. Sensors tested from their actual body position, so two sensors
# at different pos_cm on the same face cover different spatial volumes.
R_DETECT_UW = 1.0   # [m]  Ping2 short-range detection radius
R_DETECT_SC = 6.0   # [m]  radar/lidar short-range detection radius

# ──────────────────────────────────────────────────────────────────────────────
#  SENSOR CONFIGURATION GUIDE
# ──────────────────────────────────────────────────────────────────────────────
# face      : "front" | "back" | "top" | "bottom" | "left" | "right"
#
# pos_cm    : offset [cm] from face centre
#               front/back  → z-offset (up from hemisphere centre); optional, default 0
#               top/bottom/left/right → x-offset along body longitudinal axis
#
# tilt_deg  : tilt the sensor direction away from the face normal [deg]; optional, default 0
#             Positive = clockwise when viewed from outside the face looking in.
#             Per face, positive tilts toward:
#               front  → drone left  (+y)     back   → drone right (-y)
#               left   → backward   (-x)     right  → forward    (+x)
#               top    → drone right (-y)     bottom → drone left  (+y)
#             Use negative to tilt the other way.
#
# --- circular cone FOV ---
# beamwidth : total cone angle [deg]  —  half-angle = beamwidth / 2
#
# --- rectangular pyramid FOV ---
# hfov      : full horizontal angle [deg]  (half-angle = hfov / 2)
# vfov      : full vertical   angle [deg]  (half-angle = vfov / 2)
#
# --- 360 degree disk sonar ---
# type         : "disk360"   (no 'face' needed)
# vfov_deg     : total vertical opening angle [deg] of the sonar sheet
# z_offset_cm  : height above (+) or below (-) body centre [cm]

# ──────────────────────────────────────────────────────────────────────────────
#  UNDERWATER (UW) CONFIGURATION  —  box body, no arms
#  Body dimensions from geometry.py (L_box × W_box × H_box)
# ──────────────────────────────────────────────────────────────────────────────
UW = {
    'body_shape':      'pill',
    'R_body':          0.1,          # [m]  body radius
    'L_cyl':           0.82,         # [m]  cylinder section length
    'show_thrusters':  True,         # draw 4 horizontal thrusters at CoM height
    'CONE_LENGTH':     1,         # [m]  visualisation length of each sensor cone
    'sensors': [
        {"name": "Ping2 Sonar",  "face": "front",                    "beamwidth": 30},
        {"name": "Ping2 Sonar",  "face": "left",   "pos_cm":   0,    "beamwidth": 30},
        {"name": "Ping2 Sonar",  "face": "right",  "pos_cm":   0,    "beamwidth": 30},
        {"name": "Ping2 Sonar",  "face": "top",    "pos_cm": -18.5,  "beamwidth": 30},
        {"name": "Ping2 Sonar",  "face": "top",    "pos_cm":  18.5,  "beamwidth": 30},
        {"name": "Ping2 Sonar",  "face": "bottom", "pos_cm": -18.5,  "beamwidth": 30},
        {"name": "Ping2 Sonar",  "face": "bottom", "pos_cm":  18.5,  "beamwidth": 30},
        {"name": "360 Disk Sonar", "type": "disk360", "vfov_deg": 20, "z_offset_cm": 0},
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
#  AERIAL (SC) CONFIGURATION  —  pill body, with arms
# ──────────────────────────────────────────────────────────────────────────────
SC = {
    'body_shape':    'pill',
    'R_body':        0.1,         # [m]  body radius (cylinder + hemisphere)
    'L_cyl':         0.55,        # [m]  cylinder section length
    'L_arm':         0.70,        # [m]  arm length (body centre → rotor centre)
    'R_rotor':       _ap_D / 2,   # [m]  rotor disc radius (from aerial_props D)
    'ARM_ANGLE_DEG': 30.5,        # [deg] angle from X-axis to one arm
    'show_arms':     True,
    'CONE_LENGTH':   0.32,        # [m]  visualisation length of each sensor cone
    'sensors': [
        {"name": "Lidar",  "face": "front",  "hfov": 70, "vfov": 70},
        {"name": "IWR1843AOP radar",  "face": "back",  "hfov": 140, "vfov": 140},
        {"name": "IWR1843AOP radar",  "face": "top",  "pos_cm":   0, "hfov": 140, "vfov": 140},
        {"name": "IWR1843AOP radar",  "face": "bottom",   "pos_cm":   0, "hfov": 140, "vfov": 140},
        {"name": "IWR1843AOP radar",  "face": "left",   "pos_cm":   0,  "hfov": 140, "vfov": 140},
        {"name": "IWR1843AOP radar",  "face": "right",    "pos_cm":   0, "hfov": 140, "vfov": 140},
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
#  INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────
_FACE_DIR = {
    "front":  np.array([ 1.,  0.,  0.]),
    "back":   np.array([-1.,  0.,  0.]),
    "top":    np.array([ 0.,  0.,  1.]),
    "bottom": np.array([ 0.,  0., -1.]),
    "left":   np.array([ 0.,  1.,  0.]),
    "right":  np.array([ 0., -1.,  0.]),
}

# Horizontal and vertical axes as seen looking outward from each face.
_FACE_AXES = {
    "front":  (np.array([0.,  1., 0.]), np.array([0., 0., 1.])),  # h=left,    v=up
    "back":   (np.array([0., -1., 0.]), np.array([0., 0., 1.])),  # h=right,   v=up
    "top":    (np.array([0.,  1., 0.]), np.array([1., 0., 0.])),  # h=left,    v=forward
    "bottom": (np.array([0.,  1., 0.]), np.array([-1., 0., 0.])), # h=left,    v=backward
    "left":   (np.array([1.,  0., 0.]), np.array([0., 0., 1.])),  # h=forward, v=up
    "right":  (np.array([-1., 0., 0.]), np.array([0., 0., 1.])),  # h=back,    v=up
}

def _parse(sensor_list, cfg):
    shape = cfg['body_shape']
    if shape == 'pill':
        R_body = cfg['R_body'];  L_cyl = cfg['L_cyl']
        half_x = L_cyl / 2
        face_y = {'top': 0.,      'bottom': 0.,       'left':  R_body, 'right': -R_body}
        face_z = {'top': R_body,  'bottom': -R_body,  'left':  0.,     'right':  0.    }
    else:  # box
        half_x = cfg['L_x'] / 2
        hy = cfg['W_y'] / 2;  hz = cfg['H_z'] / 2
        face_y = {'top': 0.,  'bottom': 0.,   'left':  hy,  'right': -hy}
        face_z = {'top': hz,  'bottom': -hz,  'left':  0.,  'right':  0.}

    out = []
    for s in sensor_list:
        # ── 360° disk sonar — special type, no face ───────────────────────────
        if s.get("type") == "disk360":
            z_off  = s.get("z_offset_cm", 0.0) / 100.0
            pos    = np.array([0., 0., z_off])
            half_v = np.radians(s.get("vfov_deg", 10.0) / 2.0)
            out.append({"name": s["name"], "pos": pos,
                        "fov_type": "disk360",
                        "half_v": half_v,
                        "vfov_deg": s.get("vfov_deg", 10.0),
                        "dir": np.array([1., 0., 0.]),   # unused, kept for API
                        "face": "—"})
            continue

        face = s["face"]
        d    = _FACE_DIR[face].copy()

        if face in ("front", "back"):
            x_sign = 1. if face == "front" else -1.
            pos_z  = s.get("pos_cm", 0) / 100.0   # z-offset from hemisphere centre
            pos    = np.array([x_sign * half_x, 0., pos_z])
        else:
            x   = s["pos_cm"] / 100.0
            pos = np.array([x, face_y[face], face_z[face]])

        h_ax, v_ax = _FACE_AXES[face]
        h_ax = h_ax.copy();  v_ax = v_ax.copy()

        # Tilt: rotate d (and h_ax for pyramid sensors) around v_ax.
        # Positive tilt_deg = clockwise viewed from outside the face.
        tilt_deg = s.get("tilt_deg", 0.0)
        if tilt_deg != 0.0:
            tr = np.radians(tilt_deg)
            ct, st = np.cos(tr), np.sin(tr)
            d    = d    * ct + np.cross(v_ax, d)    * st
            h_ax = h_ax * ct + np.cross(v_ax, h_ax) * st
            d    = d    / np.linalg.norm(d)
            h_ax = h_ax / np.linalg.norm(h_ax)

        if "beamwidth" in s:
            fov = {"fov_type": "cone",
                   "half_angle": np.radians(s["beamwidth"] / 2),
                   "beamwidth": s["beamwidth"]}
        else:
            fov = {"fov_type": "pyramid",
                   "half_h": np.radians(s["hfov"] / 2),
                   "half_v": np.radians(s["vfov"] / 2),
                   "h_axis": h_ax, "v_axis": v_ax,
                   "hfov": s["hfov"], "vfov": s["vfov"]}

        out.append({"name": s["name"], "pos": pos, "dir": d, "face": face, **fov})
    return out

def _perp_basis(d):
    ref = np.array([0., 0., 1.]) if abs(d[2]) < 0.9 else np.array([1., 0., 0.])
    e1  = np.cross(d, ref);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(d, e1)
    return e1, e2

def _draw_cone(ax, pos, direction, half_angle, length, color, alpha=0.22, n_phi=40):
    d  = direction / np.linalg.norm(direction)
    e1, e2 = _perp_basis(d)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    rim = (pos[:, None]
           + length * d[:, None]
           + length * np.tan(half_angle) * (np.outer(e1, np.cos(phi))
                                            + np.outer(e2, np.sin(phi))))
    verts = [[pos.tolist(), rim[:, i].tolist(), rim[:, (i + 1) % n_phi].tolist()]
             for i in range(n_phi)]
    ax.add_collection3d(Poly3DCollection(verts, alpha=alpha,
                                         facecolor=color, edgecolor='none'))
    rc = np.hstack([rim, rim[:, :1]])
    ax.plot(rc[0], rc[1], rc[2], color=color, lw=1.2, alpha=0.7)
    for i in range(0, n_phi, n_phi // 4):
        ax.plot([pos[0], rim[0, i]], [pos[1], rim[1, i]], [pos[2], rim[2, i]],
                color=color, lw=0.8, alpha=0.55)

def _draw_pyramid(ax, pos, direction, h_axis, v_axis, half_h, half_v, length,
                  color, alpha=0.22):
    d = direction / np.linalg.norm(direction)
    th, tv = np.tan(half_h), np.tan(half_v)
    # 4 base corners: top-left, top-right, bottom-right, bottom-left
    c = [pos + length * (d + sh * th * h_axis + sv * tv * v_axis)
         for sh, sv in [(+1, +1), (-1, +1), (-1, -1), (+1, -1)]]
    # 4 side faces (triangles from apex)
    for i in range(4):
        ax.add_collection3d(Poly3DCollection(
            [[pos.tolist(), c[i].tolist(), c[(i + 1) % 4].tolist()]],
            alpha=alpha, facecolor=color, edgecolor='none'))
    # base face (quad)
    ax.add_collection3d(Poly3DCollection(
        [[ci.tolist() for ci in c]],
        alpha=alpha * 0.5, facecolor=color, edgecolor='none'))
    # rim + edge lines
    rim = np.array(c + [c[0]])
    ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], color=color, lw=1.2, alpha=0.7)
    for ci in c:
        ax.plot([pos[0], ci[0]], [pos[1], ci[1]], [pos[2], ci[2]],
                color=color, lw=0.8, alpha=0.55)

def _draw_disk360(ax, pos, half_v, length, color, alpha=0.25, n=80):
    """Draw a flat annular wedge representing a 360° disk sonar.
    Shows the upper and lower extent rings + transparent fill between them."""
    phi = np.linspace(0, 2 * np.pi, n)
    dz  = length * np.tan(half_v)   # vertical extent at range = length
    for sign in (+1, -1):
        zr = pos[2] + sign * dz
        xr = pos[0] + length * np.cos(phi)
        yr = pos[1] + length * np.sin(phi)
        ax.plot(xr, yr, np.full_like(phi, zr), color=color, lw=1.2, alpha=0.7)
    # filled wedge faces (top and bottom disc)
    for sign in (+1, -1):
        zr   = pos[2] + sign * dz
        ring = np.column_stack([pos[0] + length * np.cos(phi),
                                pos[1] + length * np.sin(phi),
                                np.full(n, zr)])
        ax.add_collection3d(Poly3DCollection([ring.tolist()],
                                              alpha=alpha, facecolor=color, edgecolor='none'))
    # vertical lines every 90° to show wedge depth
    for ang in np.linspace(0, 2 * np.pi, 5)[:-1]:
        xe = pos[0] + length * np.cos(ang)
        ye = pos[1] + length * np.sin(ang)
        ax.plot([xe, xe], [ye, ye], [pos[2] - dz, pos[2] + dz],
                color=color, lw=0.8, alpha=0.55)

def _rotor_disc(ax, cx, cy, r, cz=0., color='dimgray', alpha=0.30):
    phi  = np.linspace(0, 2 * np.pi, 50)
    vx   = cx + r * np.cos(phi)
    vy   = cy + r * np.sin(phi)
    vz   = np.full_like(phi, cz)
    ring = np.column_stack([vx, vy, vz])
    ax.add_collection3d(Poly3DCollection([ring], alpha=alpha,
                                          facecolor=color, edgecolor=color, lw=0.6))

def _draw_thruster(ax, pos, direction, r=0.04, color='steelblue', alpha=0.65):
    """Small disc (face perpendicular to thrust) + arrow showing thrust direction."""
    d  = direction / np.linalg.norm(direction)
    e1, e2 = _perp_basis(d)
    phi  = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    rim  = pos[:, None] + r * (np.outer(e1, np.cos(phi)) + np.outer(e2, np.sin(phi)))
    ax.add_collection3d(Poly3DCollection([rim.T.tolist()], alpha=alpha,
                                          facecolor=color, edgecolor=color, lw=0.6))
    tip = pos + d * r * 3.0
    ax.plot([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]],
            color=color, lw=1.8, alpha=0.8)
    ax.scatter(*pos, color=color, s=18, zorder=5)

def _draw_box(ax, Lx, Wy, Hz, color='lightsteelblue', alpha=0.35):
    hx, hy, hz = Lx / 2, Wy / 2, Hz / 2
    faces = [
        [[-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz]],
        [[-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz]],
        [[-hx, -hy, -hz], [ hx, -hy, -hz], [ hx, -hy,  hz], [-hx, -hy,  hz]],
        [[-hx,  hy, -hz], [ hx,  hy, -hz], [ hx,  hy,  hz], [-hx,  hy,  hz]],
        [[-hx, -hy, -hz], [-hx,  hy, -hz], [-hx,  hy,  hz], [-hx, -hy,  hz]],
        [[ hx, -hy, -hz], [ hx,  hy, -hz], [ hx,  hy,  hz], [ hx, -hy,  hz]],
    ]
    ax.add_collection3d(Poly3DCollection(faces, alpha=alpha,
                                         facecolor=color, edgecolor='steelblue', lw=0.4))

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN ANALYSIS + PLOT FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def _run(label, cfg, pts):
    R_detect = np.linalg.norm(pts[0])   # recover from first sample
    CONE_LENGTH = cfg['CONE_LENGTH']
    sensor_list = cfg['sensors']
    sl          = _parse(sensor_list, cfg)

    # ── coverage ──────────────────────────────────────────────────────────────
    # pts are points on the detection sphere (already scaled by R_detect).
    # We compute the angle from the sensor's body position to each point so
    # that sensors at different locations on the same face cover different volumes.
    def _in_fov(sb, pts):
        vec     = pts - sb["pos"]                          # (N,3) from sensor to point
        norm    = np.linalg.norm(vec, axis=1, keepdims=True)
        vec_hat = vec / (norm + 1e-9)
        if sb['fov_type'] == 'disk360':
            # detected if elevation angle from horizontal plane is within half_v
            horiz = np.sqrt(vec_hat[:, 0]**2 + vec_hat[:, 1]**2)
            elev  = np.abs(np.arctan2(np.abs(vec_hat[:, 2]), horiz + 1e-9))
            return elev <= sb["half_v"]
        if sb['fov_type'] == 'cone':
            return vec_hat @ sb["dir"] >= np.cos(sb["half_angle"])
        p_d      = vec_hat @ sb["dir"]
        in_front = p_d > 0
        ang_h    = np.where(in_front, np.abs(np.arctan2(vec_hat @ sb["h_axis"], p_d)), np.inf)
        ang_v    = np.where(in_front, np.abs(np.arctan2(vec_hat @ sb["v_axis"], p_d)), np.inf)
        return (ang_h <= sb["half_h"]) & (ang_v <= sb["half_v"])

    covered = np.zeros(len(pts), dtype=bool)
    for sb in sl:
        covered |= _in_fov(sb, pts)
    coverage_pct = 100.0 * covered.sum() / len(pts)

    print(f"\n{'═'*60}")
    print(f"  {label}  —  Total coverage : {coverage_pct:.1f}%")
    print(f"{'═'*60}")
    print(f"  {'Sensor':<16}  {'Face':<7}  {'FOV':<20}  {'Solo':>6}")
    print(f"  {'-'*54}")
    for sb in sl:
        solo = 100.0 * _in_fov(sb, pts).sum() / len(pts)
        if sb['fov_type'] == 'disk360':
            fov_str = f"360° disk  V {sb['vfov_deg']}°"
        elif sb['fov_type'] == 'cone':
            fov_str = f"BW {sb['beamwidth']}°"
        else:
            fov_str = f"H {sb['hfov']}° × V {sb['vfov']}°"
        print(f"  {sb['name']:<16}  {sb['face']:<7}  {fov_str:<20}  {solo:>5.1f}%")
    print(f"{'═'*60}\n")

    # ── colours ───────────────────────────────────────────────────────────────
    unique_names = list(dict.fromkeys(s["name"] for s in sensor_list))
    cmap         = plt.colormaps["tab10"].resampled(max(len(unique_names), 1))
    name_color   = {name: cmap(i) for i, name in enumerate(unique_names)}

    # ── plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection='3d')

    shape = cfg['body_shape']

    if shape == 'pill':
        R_body = cfg['R_body'];  L_cyl = cfg['L_cyl']
        _u  = np.linspace(0, 2 * np.pi, 50)
        _ph = np.linspace(0, np.pi / 2, 20)
        _xc, _uc = np.meshgrid(np.linspace(-L_cyl / 2, L_cyl / 2, 8), _u)
        ax.plot_surface(_xc, R_body * np.cos(_uc), R_body * np.sin(_uc),
                        color='lightsteelblue', alpha=0.35, edgecolor='none')
        _PH, _UC = np.meshgrid(_ph, _u)
        ax.plot_surface( L_cyl / 2 + R_body * np.cos(_PH),
                         R_body * np.sin(_PH) * np.cos(_UC),
                         R_body * np.sin(_PH) * np.sin(_UC),
                         color='lightsteelblue', alpha=0.35, edgecolor='none')
        ax.plot_surface(-L_cyl / 2 - R_body * np.cos(_PH),
                         R_body * np.sin(_PH) * np.cos(_UC),
                         R_body * np.sin(_PH) * np.sin(_UC),
                         color='lightsteelblue', alpha=0.35, edgecolor='none')
        body_extent = L_cyl / 2 + R_body

    else:  # box
        _draw_box(ax, cfg['L_x'], cfg['W_y'], cfg['H_z'])
        body_extent = max(cfg['L_x'], cfg['W_y'], cfg['H_z']) / 2

    if cfg.get('show_arms'):
        R_body        = cfg['R_body']
        L_arm         = cfg['L_arm']
        R_rotor       = cfg['R_rotor']
        ARM_ANGLE_DEG = cfg['ARM_ANGLE_DEG']
        for a_deg in [ARM_ANGLE_DEG, 180 - ARM_ANGLE_DEG,
                      180 + ARM_ANGLE_DEG, 360 - ARM_ANGLE_DEG]:
            a_rad   = np.radians(a_deg)
            arm_dir = np.array([np.cos(a_rad), np.sin(a_rad), 0.])
            t_surf  = R_body / abs(np.sin(a_rad))
            start   = t_surf * arm_dir
            tip     = L_arm  * arm_dir
            ax.plot([start[0], tip[0]], [start[1], tip[1]], [R_body, R_body],
                    color='dimgray', lw=2.2, zorder=3)
            _rotor_disc(ax, tip[0], tip[1], R_rotor, cz=R_body)
        arm_extent = cfg['L_arm'] + cfg['R_rotor']
    else:
        arm_extent = 0.0

    if cfg.get('show_thrusters') and cfg['body_shape'] == 'pill':
        R_body = cfg['R_body'];  L_cyl = cfg['L_cyl']
        cg, sg = np.cos(np.radians(45)), np.sin(np.radians(45))
        # 4 thrusters at pill corners at z=0 (CoM height), 45° inward
        # layout mirrors quadcopterUW horizontal thruster mixing matrix
        for px, py, dx, dy in [
            ( L_cyl/2,  R_body,  cg, -sg),   # front-left
            ( L_cyl/2, -R_body,  cg,  sg),   # front-right
            (-L_cyl/2,  R_body, -cg, -sg),   # rear-left
            (-L_cyl/2, -R_body, -cg,  sg),   # rear-right
        ]:
            _draw_thruster(ax, np.array([px, py, 0.]), np.array([dx, dy, 0.]))

    # sensor shapes
    legend_done = set()
    legend_hdls = []
    for sb in sl:
        col = name_color[sb["name"]]
        if sb['fov_type'] == 'disk360':
            _draw_disk360(ax, sb["pos"], sb["half_v"], CONE_LENGTH, col)
        elif sb['fov_type'] == 'cone':
            _draw_cone(ax, sb["pos"], sb["dir"], sb["half_angle"], CONE_LENGTH, col)
        else:
            _draw_pyramid(ax, sb["pos"], sb["dir"], sb["h_axis"], sb["v_axis"],
                          sb["half_h"], sb["half_v"], CONE_LENGTH, col)
        ax.scatter(*sb["pos"], color=col, s=25, zorder=6)
        if sb["name"] not in legend_done:
            legend_hdls.append(mpatches.Patch(color=col, label=sb["name"], alpha=0.75))
            legend_done.add(sb["name"])

    _half = max(body_extent + CONE_LENGTH, arm_extent) * 1.15
    ax.set_xlim(-_half, _half);  ax.set_ylim(-_half, _half);  ax.set_zlim(-_half, _half)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X  (forward) [m]")
    ax.set_ylabel("Y  (left)    [m]")
    ax.set_zlabel("Z  (up)      [m]")
    ax.set_title(f"{label}  —  Spatial coverage  {coverage_pct:.1f}%  (R_detect = {R_detect:.1f} m)",
                 fontsize=13, fontweight='bold')
    if legend_hdls:
        ax.legend(handles=legend_hdls, loc='upper left', fontsize=10)

    plt.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
#  RUN
# ──────────────────────────────────────────────────────────────────────────────
_rng      = np.random.default_rng(42)
_pts_unit = _rng.standard_normal((N_MC, 3))
_pts_unit /= np.linalg.norm(_pts_unit, axis=1, keepdims=True)

# Scale unit vectors to the detection radius so sensor position matters
_run("Underwater (UW)", UW, _pts_unit * R_DETECT_UW)
_run("Aerial (SC)",     SC, _pts_unit * R_DETECT_SC)

plt.show()
