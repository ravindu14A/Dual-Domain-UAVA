"""
Top-view geometry of a telescoping X-quad on a pill-shaped body.

Two arm pairs (A and B), each pair is two parallel tubes offset by 2w
(one tube width each side of the pair centre-line) so they slide past each other.

Pair A  –  Arm 1 (upper-right, angle +alpha) and Arm 3 (lower-left, angle 180+alpha)
Pair B  –  Arm 2 (lower-right, angle -alpha) and Arm 4 (upper-left, angle 180-alpha)

For each tube:
  t_exit  > 0  : body boundary on the prop side
  t_entry < 0  : body boundary on the retraction side (other end of chord)
  chord        = t_exit - t_entry
  L_tube       = (R_prop + gap) + overlap   (arm extension + structural overlap inside body)
  hanging_out  = (R_prop + gap + overlap) - chord   (positive => sticks out when fully retracted)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider

# ── pill helpers ──────────────────────────────────────────────────────────────

def pill_outline(R, L, n=400):
    th_r = np.linspace(-np.pi/2,  np.pi/2,  n//4)
    th_l = np.linspace( np.pi/2, 3*np.pi/2, n//4)
    xs = np.concatenate([ L/2 + R*np.cos(th_r),
                           np.linspace( L/2, -L/2, n//4),
                          -L/2 + R*np.cos(th_l),
                           np.linspace(-L/2,  L/2, n//4)])
    ys = np.concatenate([ R*np.sin(th_r),
                           np.full(n//4,  R),
                           R*np.sin(th_l),
                           np.full(n//4, -R)])
    return xs, ys

def inside_pill(x, y, R, L):
    if -L/2 <= x <= L/2:
        return y**2 <= R**2 + 1e-9
    elif x > L/2:
        return (x-L/2)**2 + y**2 <= R**2 + 1e-9
    else:
        return (x+L/2)**2 + y**2 <= R**2 + 1e-9

# ── arm geometry ──────────────────────────────────────────────────────────────

def body_crossings(alpha, w_off, R, L):
    """Intersect arm axis with pill boundary.

    Axis: P(t) = w_off*(-sin(a), cos(a))  +  t*(cos(a), sin(a))
    Returns (t_exit, t_entry):
      t_exit  > 0  prop-side body wall
      t_entry < 0  retraction-side body wall
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ox = -w_off * sa
    oy =  w_off * ca
    search = 1.5 * (R + L/2 + 1.0)
    ts = np.linspace(-search, search, 30000)
    inside = np.array([inside_pill(ox + t*ca, oy + t*sa, R, L) for t in ts])
    idx = np.where(np.diff(inside.astype(int)) != 0)[0]
    if len(idx) < 2:
        return None, None
    crossings = (ts[idx] + ts[idx+1]) / 2
    pos = crossings[crossings > 0]
    neg = crossings[crossings < 0]
    if not len(pos) or not len(neg):
        return None, None
    return float(pos.min()), float(neg.max())   # t_exit, t_entry


def arm_geom(alpha, w_off, R, L, R_prop, gap, overlap):
    """Compute all derived quantities for one arm tube."""
    t_exit, t_entry = body_crossings(alpha, w_off, R, L)
    if t_exit is None:
        return None
    ca, sa  = np.cos(alpha), np.sin(alpha)
    ox, oy  = -w_off*sa, w_off*ca         # axis origin (offset point)
    arm_ext = R_prop + gap                 # minimum arm sticking out past body
    t_prop  = t_exit + arm_ext            # prop centre (on axis)
    chord   = t_exit - t_entry            # body chord along arm direction
    L_tube  = arm_ext + overlap            # tube length
    # max retraction: inner end slides to t_entry (far body wall)
    t_ret_outer = t_entry + L_tube        # outer end position when fully retracted
    hanging_out = t_ret_outer - t_exit    # = arm_ext + overlap - chord
    return dict(ox=ox, oy=oy, ca=ca, sa=sa,
                t_exit=t_exit, t_entry=t_entry,
                t_prop=t_prop, chord=chord,
                L_tube=L_tube, t_ret_outer=t_ret_outer,
                hanging_out=hanging_out, arm_ext=arm_ext)


# ── drawing helpers ───────────────────────────────────────────────────────────

def draw_tube(ax, g, w, t0, t1, **kw):
    """Draw arm tube as a filled polygon from t0 to t1 along arm axis."""
    ox, oy, ca, sa = g['ox'], g['oy'], g['ca'], g['sa']
    hw = w / 2
    px, py = -sa, ca                      # perpendicular unit vector
    c = np.array([
        [ox + t0*ca + hw*px, oy + t0*sa + hw*py],
        [ox + t0*ca - hw*px, oy + t0*sa - hw*py],
        [ox + t1*ca - hw*px, oy + t1*sa - hw*py],
        [ox + t1*ca + hw*px, oy + t1*sa + hw*py],
    ])
    ax.add_patch(plt.Polygon(c, **kw))


def draw(R, L, w, alpha_deg, R_prop, gap, overlap):
    ax.cla()
    alpha = np.radians(alpha_deg)

    # Pill body
    xs, ys = pill_outline(R, L)
    ax.fill(xs, ys, color='#dce8f5', zorder=1)
    ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]),
            color='steelblue', lw=1.8, zorder=2)

    # 4 arm tubes.
    # Each arm's perpendicular offset is +w in its own perp direction.
    # Pair A: angles +alpha and pi+alpha  (offset directions are antiparallel → 2w separation)
    # Pair B: angles -alpha and pi-alpha
    ARMS = [
        ( alpha,        +w, '#e63946', '#ffb3ba', 'A1 (+α)'),
        (-alpha,        +w, '#2a9d8f', '#b2dfdb', 'B1 (-α)'),
        (np.pi+alpha,   +w, '#c1121f', '#ff6b6b', 'A3 (180+α)'),
        (np.pi-alpha,   +w, '#1b7a6e', '#80cbc4', 'B2 (180-α)'),
    ]

    info = []
    any_valid = False

    for (angle, w_off, col_ext, col_ret, label) in ARMS:
        g = arm_geom(angle, w_off, R, L, R_prop, gap, overlap)
        if g is None:
            info.append(f'{label}: arm outside body — check α/R/L')
            continue
        any_valid = True

        ox, oy, ca, sa = g['ox'], g['oy'], g['ca'], g['sa']
        t_exit, t_entry = g['t_exit'], g['t_entry']
        t_prop          = g['t_prop']
        t_ret_outer     = g['t_ret_outer']
        chord           = g['chord']
        L_tube          = g['L_tube']
        hang            = g['hanging_out']

        # ── retracted tube (dashed outline)
        draw_tube(ax, g, w, t_entry, t_ret_outer,
                  facecolor='none', edgecolor=col_ext,
                  linewidth=1.2, linestyle='--', alpha=0.55, zorder=3)

        # ── extended tube (solid fill)
        t_inner_ext = t_exit - overlap           # inner end when extended
        draw_tube(ax, g, w, t_inner_ext, t_prop,
                  facecolor=col_ext, edgecolor='none',
                  alpha=0.75, zorder=4)

        # ── prop circle extended (solid)
        px_e = ox + t_prop * ca;  py_e = oy + t_prop * sa
        ax.add_patch(plt.Circle((px_e, py_e), R_prop,
                                color=col_ext, fill=False, lw=2.0, zorder=5))
        ax.plot(px_e, py_e, 'o', color=col_ext, ms=4, zorder=6)

        # ── prop circle retracted (dashed)
        px_r = ox + t_ret_outer * ca;  py_r = oy + t_ret_outer * sa
        ax.add_patch(plt.Circle((px_r, py_r), R_prop,
                                color=col_ext, fill=False, lw=1.2,
                                linestyle='--', alpha=0.45, zorder=3))

        # ── body wall markers
        for t_mark in (t_exit, t_entry):
            mx = ox + t_mark*ca;  my = oy + t_mark*sa
            ax.plot(mx, my, 's', color='k', ms=4, zorder=7)

        hang_str = (f'{hang*1e3:.1f} mm sticks out'
                    if hang > 1e-4 else
                    f'fully retractable ({-hang*1e3:.1f} mm margin)')
        info.append(
            f'{label}: chord={chord*100:.1f} cm  '
            f'L_tube={L_tube*100:.1f} cm  '
            f'arm_ext={g["arm_ext"]*100:.1f} cm  '
            f'→  {hang_str}'
        )

    lim = R + L/2 + R_prop + gap + 0.2
    ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25, zorder=0)
    ax.set_xlabel('X [m]');  ax.set_ylabel('Y [m]')
    ax.set_title(
        f'Top view  |  α = {alpha_deg:.1f}°  |  '
        f'R = {R:.3f} m   L = {L:.3f} m   tube w = {w*1000:.0f} mm',
        fontsize=11)

    # legend patches
    leg = [
        mpatches.Patch(color='#e63946', alpha=0.75, label='Pair A extended'),
        mpatches.Patch(color='#2a9d8f', alpha=0.75, label='Pair B extended'),
        plt.Line2D([0],[0], color='k', lw=1.2, ls='--', label='retracted (dashed)'),
        plt.Line2D([0],[0], color='k', marker='s', ms=5, ls='none', label='body wall crossing'),
    ]
    ax.legend(handles=leg, loc='upper right', fontsize=8)

    # info box
    ax.text(0.01, 0.01, '\n'.join(info),
            transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

    fig.canvas.draw_idle()


# ── figure layout ─────────────────────────────────────────────────────────────

R0, L0      = 0.20, 0.65
w0          = 0.04
alpha0      = 45.0
Rp0         = 0.15
gap0        = 0.05
overlap0    = 0.10

fig, ax = plt.subplots(figsize=(11, 9))
plt.subplots_adjust(bottom=0.40)

slider_defs = [
    ('Pill R [m]',     0.05, 0.50, R0,       0.005, 0.33),
    ('Pill L [m]',     0.10, 2.00, L0,       0.01,  0.28),
    ('Tube w [m]',     0.01, 0.12, w0,       0.005, 0.23),
    ('α [deg]',        5.0,  85.0, alpha0,   0.5,   0.18),
    ('Prop R [m]',     0.05, 0.50, Rp0,      0.005, 0.13),
    ('Gap [m]',        0.01, 0.20, gap0,     0.005, 0.08),
    ('Overlap [m]',    0.02, 0.40, overlap0, 0.005, 0.03),
]

sliders = []
for label, vmin, vmax, vinit, vstep, ypos in slider_defs:
    ax_sl = fig.add_axes([0.12, ypos, 0.78, 0.025])
    sl = Slider(ax_sl, label, vmin, vmax, valinit=vinit, valstep=vstep)
    sliders.append(sl)

def _update(_):
    draw(*[s.val for s in sliders])

for sl in sliders:
    sl.on_changed(_update)

draw(R0, L0, w0, alpha0, Rp0, gap0, overlap0)
plt.show()
