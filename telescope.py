"""
Top-view of telescoping X-quad on a pill body.

3-D GEOMETRY
────────────
Pill long axis = X.  Two arm pairs at angle ±alpha from X:

  Pair A  (z = 0)   diagonal at +alpha:  Arm A1 (upper-right)  +  Arm A3 (lower-left)
  Pair B  (z = w)   diagonal at −alpha:  Arm B1 (lower-right)  +  Arm B2 (upper-left)

Within each pair the two tubes are OPPOSITE props on the SAME horizontal plane.
They are offset ±w/2 from the pair centre-line (which passes through the body
centre), so the centre-of-pair passes through the origin and the tubes can slide
past each other along the diagonal.

Pair B sits one arm-width (w) above Pair A in Z so the two diagonals cross inside
the body without colliding.  Body cross-section schematic (looking along X):

      z=w  ┌────┐ ┌────┐   ← Pair B tubes (side by side, touching, w×w each)
           │ B1 │ │ B2 │
      z=0  ├────┤ ├────┤   ← Pair A tubes
           │ A1 │ │ A3 │
           └────┘ └────┘
           ←  2w  →

TUBE LENGTH / RETRACTION
────────────────────────
  t_exit   > 0  body wall on prop side (along arm axis from offset origin)
  t_entry  < 0  body wall on retraction side
  chord    = t_exit − t_entry

  arm_ext  = R_prop + gap          prop centre must be this far past body exit
  L_tube   = arm_ext + overlap     tube length (arm ext + structural overlap inside body)

  max retraction: inner end slides to t_entry (far body wall)
  prop centre then at  t_ret = t_entry + L_tube

  hanging_out = (t_ret − t_exit) + R_prop = 2·R_prop + gap + overlap − chord
              = L1 − L2 + overlap + R_prop  where L1 = R_prop+gap, L2 = chord
              > 0  →  prop tip still sticks out this much when fully retracted
              < 0  →  arm fully retractable inside body (margin = |hanging_out|)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════════
#  PARAMETERS — edit these before running
# ═══════════════════════════════════════════════════════════
R        = 0.1   # pill hemisphere radius [m]
L        = 0.55  # pill cylinder length [m]
w        = 0.04  # arm tube width [m]
R_prop   = 0.30  # propeller radius [m]
gap      = 0.02  # clearance between prop disk edge and body [m]
overlap  = 0.05  # structural overlap inside body when extended [m]
gap_prop = 0.04  # min horizontal clearance between adjacent prop disk edges (aero interference) [m]
# ═══════════════════════════════════════════════════════════

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
        return (x - L/2)**2 + y**2 <= R**2 + 1e-9
    else:
        return (x + L/2)**2 + y**2 <= R**2 + 1e-9


# ── arm geometry ──────────────────────────────────────────────────────────────

def body_crossings(alpha, w_off, R, L, n=40000):
    """Find t_exit (>0) and t_entry (<0) for arm axis.

    Axis:  P(t) = w_off·(-sin α, cos α)  +  t·(cos α, sin α)
    w_off = ±w/2  (half-tube-width offset from pair centre-line)
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ox, oy = -w_off * sa,  w_off * ca
    span   = 1.5 * (R + L / 2 + 1.0)
    ts     = np.linspace(-span, span, n)
    inside = np.fromiter(
        (inside_pill(ox + t*ca, oy + t*sa, R, L) for t in ts), bool, n)
    edges  = np.where(np.diff(inside.astype(int)) != 0)[0]
    if len(edges) < 2:
        return None, None
    crossings = (ts[edges] + ts[edges + 1]) / 2
    pos = crossings[crossings > 0]
    neg = crossings[crossings < 0]
    if not len(pos) or not len(neg):
        return None, None
    return float(pos.min()), float(neg.max())   # t_exit, t_entry


def sdf_pill(px, py, R, L):
    """Signed distance from (px,py) to pill boundary. Negative = inside."""
    rx = max(abs(px) - L / 2, 0.0)
    ry = abs(py)
    return np.sqrt(rx*rx + ry*ry) - R


def find_t_prop(ox, oy, ca, sa, t_exit, R, L, R_prop, gap):
    """Find smallest t past t_exit where prop circle clears body by gap.

    Condition: sdf_pill(prop_centre) >= R_prop + gap
    (nearest point on prop disk is then exactly gap from body surface)
    """
    required = R_prop + gap
    ts = np.linspace(t_exit, t_exit + 3*(R + L/2 + R_prop), 10000)
    for t in ts:
        if sdf_pill(ox + t*ca, oy + t*sa, R, L) >= required:
            return float(t)
    return float(ts[-1])   # fallback


def arm_geom(alpha, w_off, R, L, R_prop, gap, overlap):
    t_exit, t_entry = body_crossings(alpha, w_off, R, L)
    if t_exit is None:
        return None
    ca, sa  = np.cos(alpha), np.sin(alpha)
    ox, oy  = -w_off * sa, w_off * ca
    t_prop  = find_t_prop(ox, oy, ca, sa, t_exit, R, L, R_prop, gap)
    arm_ext = t_prop - t_exit          # true clearance-corrected arm extension
    chord   = t_exit - t_entry
    L_tube  = arm_ext + overlap
    L1      = t_prop   # centre → prop centre
    L2      = t_exit   # centre → body exit (prop side); far wall at -L2 (symmetric)
    # retraction limit: inner end hits far wall at -L2
    # prop centre then at -L2 + L_tube = L1 - 2*L2 + overlap
    # body-edge constraint: prop centre cannot go past exit (t_ret >= L2)
    t_ret   = max(L1 - 2*L2 + overlap, L2)
    hang    = t_ret - L2 + R_prop     # = max(L1 - 3*L2 + overlap, 0) + R_prop
    return dict(ox=ox, oy=oy, ca=ca, sa=sa,
                t_exit=t_exit, t_entry=t_entry, t_prop=t_prop,
                chord=chord, L_tube=L_tube, t_ret=t_ret, hang=hang,
                arm_ext=arm_ext)


# ── drawing helpers ───────────────────────────────────────────────────────────

def tube_polygon(g, w, t0, t1):
    """Corner array for an arm tube rectangle from t0 to t1."""
    ox, oy, ca, sa = g['ox'], g['oy'], g['ca'], g['sa']
    hw = w / 2
    nx, ny = -sa, ca              # perpendicular unit vector
    return np.array([
        [ox + t0*ca + hw*nx, oy + t0*sa + hw*ny],
        [ox + t0*ca - hw*nx, oy + t0*sa - hw*ny],
        [ox + t1*ca - hw*nx, oy + t1*sa - hw*ny],
        [ox + t1*ca + hw*nx, oy + t1*sa + hw*ny],
    ])


def add_tube(ax, g, w, t0, t1, **kw):
    ax.add_patch(plt.Polygon(tube_polygon(g, w, t0, t1), **kw))


# ── main draw ─────────────────────────────────────────────────────────────────

# Pair A = red/orange,  Pair B = teal/green
PAIR_COLS = {
    'A': ('#c1121f', '#ffb3ba'),   # (extended fill, retracted edge)
    'B': ('#1b6ca8', '#90caf9'),
}

def draw(R, L, w, alpha_deg, R_prop, gap, overlap):
    ax.cla()
    alpha = np.radians(alpha_deg)

    # ── pill ──
    xs, ys = pill_outline(R, L)
    ax.fill(xs, ys, color='#dce8f5', zorder=1)
    ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]),
            color='steelblue', lw=2, zorder=2)
    
    # centre cross
    ax.plot(0, 0, '+', color='grey', ms=10, mew=1.2, zorder=3)

    # ── arm pair centre-lines (for reference) ──
    cl = 1.2 * (R + L/2)
    for ang, col in [(alpha, PAIR_COLS['A'][0]), (-alpha, PAIR_COLS['B'][0])]:
        ax.plot([-cl*np.cos(ang), cl*np.cos(ang)],
                [-cl*np.sin(ang), cl*np.sin(ang)],
                color=col, lw=0.6, ls=':', alpha=0.4, zorder=2)

    # ── horizontal axis reference + alpha label ──
    ax.annotate('', xy=(cl, 0), xytext=(-cl, 0),
                arrowprops=dict(arrowstyle='->', color='grey', lw=0.8))
    ax.text(cl * 0.98, 0.012, 'X (horizontal)', fontsize=7,
            color='grey', ha='right', va='bottom')

    arc_r = min(R * 0.6, 0.08)   # arc radius scaled to body
    arc_th = np.linspace(0, alpha, 60)
    ax.plot(arc_r * np.cos(arc_th), arc_r * np.sin(arc_th),
            color=PAIR_COLS['A'][0], lw=1.5, zorder=5)
    # label midpoint of arc
    mid = alpha / 2
    ax.text(arc_r * 1.25 * np.cos(mid), arc_r * 1.25 * np.sin(mid),
            f'α={alpha_deg:.0f}°', fontsize=8,
            color=PAIR_COLS['A'][0], ha='center', va='center')

    # ── 4 arm tubes ──
    # w_off = +w/2 or -w/2 (half-width either side of pair centre-line)
    # Each pair: two OPPOSITE arms (props pointing away from each other).
    # Both arms in a pair use offset +w/2 in their own perp direction — this
    # puts them on parallel lines w apart, with the pair midline through origin.
    arms = [
        # (angle,              w_off,  pair, label)
        ( alpha,              +w/2,  'A', 'A1 upper-right'),
        ( alpha + np.pi,      +w/2,  'A', 'A3 lower-left '),  # opposite prop ✓
        (-alpha,              +w/2,  'B', 'B1 lower-right'),
        (-alpha + np.pi,      +w/2,  'B', 'B2 upper-left '),  # opposite prop ✓
    ]

    info_lines = []

    for angle, w_off, pair, label in arms:
        col_ext, col_ret = PAIR_COLS[pair]
        g = arm_geom(angle, w_off, R, L, R_prop, gap, overlap)
        if g is None:
            info_lines.append(f'{label}: no valid crossing — adjust R/L/α')
            continue

        ox, oy, ca, sa = g['ox'], g['oy'], g['ca'], g['sa']
        t_exit, t_entry = g['t_exit'], g['t_entry']
        t_prop, t_ret   = g['t_prop'], g['t_ret']
        chord, hang     = g['chord'], g['hang']
        L_tube          = g['L_tube']

        # retracted outline (dashed) — tube is fixed length, slides as one piece
        t_ret_inner = t_ret - g['L_tube']   # inner end when retracted
        add_tube(ax, g, w, t_ret_inner, t_ret,
                 facecolor='none', edgecolor=col_ret,
                 linewidth=1.1, linestyle='--', alpha=0.7, zorder=3)

        # extended fill
        add_tube(ax, g, w, t_exit - overlap, t_prop,
                 facecolor=col_ext, edgecolor='none', alpha=0.72, zorder=4)

        # prop discs
        for t_c, ls, alp in [(t_prop, '-', 0.9), (t_ret, '--', 0.35)]:
            px_c = ox + t_c * ca;  py_c = oy + t_c * sa
            ax.add_patch(plt.Circle((px_c, py_c), R_prop,
                                    color=col_ext, fill=False,
                                    lw=1.8, linestyle=ls, alpha=alp, zorder=5))
            if alp > 0.5:
                ax.plot(px_c, py_c, 'o', color=col_ext, ms=3.5, zorder=6)

        # body-wall crossing markers
        for t_m in (t_exit, t_entry):
            ax.plot(ox + t_m*ca, oy + t_m*sa, 's', color='k', ms=4, zorder=7)

        if hang > 1e-4:
            hang_str = f'hangs out {hang*1e3:.1f} mm when retracted'
        else:
            hang_str = f'fully retractable  ({-hang*1e3:.1f} mm margin)'

        info_lines.append(
            f'{label}:  chord = {chord*1e2:.1f} cm  |  '
            f'L_tube = {L_tube*1e2:.1f} cm  |  '
            f'arm_ext = {g["arm_ext"]*1e2:.1f} cm  |  {hang_str}'
        )

    # ── axes / labels ──
    lim = R + L/2 + R_prop + gap + 0.15
    ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim)
    ax.set_aspect('equal');   ax.grid(True, alpha=0.2, zorder=0)
    ax.set_xlabel('X [m]');   ax.set_ylabel('Y [m]')
    ax.set_title(
        f'Top view  |  α = {alpha_deg:.1f}°  '
        f'R = {R*1e2:.0f} cm   L = {L*1e2:.0f} cm   '
        f'tube w = {w*1e3:.0f} mm   '
        f'R_prop = {R_prop*1e2:.0f} cm   gap = {gap*1e2:.0f} cm   overlap = {overlap*1e2:.0f} cm',
        fontsize=9)

    # legend
    leg = [
        mpatches.Patch(color=PAIR_COLS['A'][0], alpha=0.8, label='Pair A extended'),
        mpatches.Patch(color=PAIR_COLS['B'][0], alpha=0.8, label='Pair B extended (plane above A)'),
        plt.Line2D([0],[0], color='grey', lw=1.1, ls='--', label='retracted'),
        plt.Line2D([0],[0], color='k', marker='s', ms=4, ls='none', label='body wall crossing'),
        plt.Line2D([0],[0], color='grey', lw=0.7, ls=':', label='pair centre-line (thru origin)'),
    ]
    ax.legend(handles=leg, loc='upper right', fontsize=7.5)

    # info box
    ax.text(0.01, 0.01, '\n'.join(info_lines),
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment='bottom', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffde7', alpha=0.85))

    # ── z-stack cross-section inset (looking along X, Y-Z plane) ──
    ax_xs.cla()
    ax_xs.set_aspect('equal')
    ax_xs.set_title('Body cross-section\n(looking along X)', fontsize=7.5)
    ax_xs.set_xlabel('lateral (Y)', fontsize=7); ax_xs.set_ylabel('Z', fontsize=7)
    ax_xs.tick_params(labelsize=6)

    # pill cross-section = circle of radius R
    theta_c = np.linspace(0, 2*np.pi, 200)
    ax_xs.fill(R*np.cos(theta_c), R*np.sin(theta_c), color='#dce8f5', zorder=1)
    ax_xs.plot(R*np.cos(theta_c), R*np.sin(theta_c), 'steelblue', lw=1.5, zorder=2)

    # Pair A (z=0): two tubes at y = -w/2 and +w/2, z bottom = -w/2
    col_A = PAIR_COLS['A'][0]
    col_B = PAIR_COLS['B'][0]
    for y_off, lbl in [(-w/2, 'A1'), (w/2, 'A3')]:
        ax_xs.add_patch(mpatches.Rectangle(
            (y_off - w/2, -w/2), w, w,
            facecolor=col_A, edgecolor='k', linewidth=0.8, alpha=0.8, zorder=3))
        ax_xs.text(y_off, 0, lbl, ha='center', va='center',
                   fontsize=6, color='white', fontweight='bold', zorder=4)

    # Pair B (z=w): two tubes at y = -w/2 and +w/2, z bottom = w/2
    for y_off, lbl in [(-w/2, 'B1'), (w/2, 'B2')]:
        ax_xs.add_patch(mpatches.Rectangle(
            (y_off - w/2, w/2), w, w,
            facecolor=col_B, edgecolor='k', linewidth=0.8, alpha=0.8, zorder=3))
        ax_xs.text(y_off, w, lbl, ha='center', va='center',
                   fontsize=6, color='white', fontweight='bold', zorder=4)

    pad = R * 0.15
    ax_xs.set_xlim(-R - pad, R + pad)
    ax_xs.set_ylim(-R - pad, R + pad)
    ax_xs.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax_xs.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax_xs.grid(True, alpha=0.2)

    fig.canvas.draw_idle()


# ── alpha sweep ───────────────────────────────────────────────────────────────

def prop_center(g):
    """XY position of prop centre for an arm geometry dict."""
    return np.array([g['ox'] + g['t_prop']*g['ca'],
                     g['oy'] + g['t_prop']*g['sa']])


def min_adj_clearance(alpha, w, R, L, R_prop, gap, overlap):
    """Minimum horizontal clearance between adjacent prop disk edges (cross-pair).

    Adjacent = one arm from Pair A (+alpha), one from Pair B (-alpha).
    Four adjacent pairs: A1-B1, A1-B2, A3-B1, A3-B2.
    Clearance = centre-to-centre distance - 2*R_prop.
    """
    arms = {
        'A1': arm_geom( alpha,          +w/2, R, L, R_prop, gap, overlap),
        'A3': arm_geom( alpha + np.pi,  +w/2, R, L, R_prop, gap, overlap),
        'B1': arm_geom(-alpha,          +w/2, R, L, R_prop, gap, overlap),
        'B2': arm_geom(-alpha + np.pi,  +w/2, R, L, R_prop, gap, overlap),
    }
    if any(v is None for v in arms.values()):
        return np.nan
    min_clr = np.inf
    for a_key in ('A1', 'A3'):
        for b_key in ('B1', 'B2'):
            d = np.linalg.norm(prop_center(arms[a_key]) - prop_center(arms[b_key]))
            min_clr = min(min_clr, d - 2*R_prop)
    return min_clr


def sweep_alpha(R, L, w, R_prop, gap, overlap, n=180):
    alphas = np.linspace(1.0, 89.0, n)
    hangs  = []
    clrs   = []
    for a_deg in alphas:
        a = np.radians(a_deg)
        g = arm_geom(a, +w/2, R, L, R_prop, gap, overlap)
        hangs.append(g['hang'] if g is not None else np.nan)
        clrs.append(min_adj_clearance(a, w, R, L, R_prop, gap, overlap))
    return alphas, np.array(hangs), np.array(clrs)

print("\nRunning alpha sweep...")
alphas, hangs, clrs = sweep_alpha(R, L, w, R_prop, gap, overlap)
valid       = ~np.isnan(hangs)
clr_valid   = ~np.isnan(clrs)
aero_ok     = clr_valid & (clrs >= gap_prop)   # satisfies aero clearance

min_hang    = np.nanmin(hangs)
# largest angle that achieves the floor AND satisfies aero clearance
at_floor    = valid & aero_ok & (hangs <= min_hang + 1e-4)
if at_floor.any():
    best_idx = int(np.where(at_floor)[0][-1])
else:
    # fall back: best overhang among aero-valid angles
    masked = np.where(aero_ok, hangs, np.nan)
    best_idx = int(np.nanargmin(masked))
best_alpha  = alphas[best_idx]
best_hang   = hangs[best_idx]
best_clr    = clrs[best_idx]

# arm length at best alpha
g_best = arm_geom(np.radians(best_alpha), +w/2, R, L, R_prop, gap, overlap)
best_arm_len = 0.0
if g_best is not None:
    bx = g_best['ox'] + g_best['t_prop'] * g_best['ca']
    by = g_best['oy'] + g_best['t_prop'] * g_best['sa']
    best_arm_len = np.sqrt(bx**2 + by**2)

print(f"\nAlpha sweep  (R={R*1e2:.0f}cm  L={L*1e2:.0f}cm  w={w*1e3:.0f}mm  "
      f"R_prop={R_prop*1e2:.0f}cm  gap={gap*1e2:.0f}cm  gap_prop={gap_prop*1e2:.0f}cm  overlap={overlap*1e2:.0f}cm)")
print(f"  Theoretical minimum overhang : {R_prop*1e3:.1f} mm  (= R_prop)")
print(f"  Best achievable overhang     : {best_hang*1e3:.2f} mm  at  alpha = {best_alpha:.1f} deg")
print(f"  Adjacent prop clearance      : {best_clr*1e3:.1f} mm  (need >= {gap_prop*1e3:.0f} mm)")
print(f"  Arm length at best alpha     : {best_arm_len*1e2:.2f} cm  (Euclidean from pill centre)")
if g_best is not None:
    L1_ax = g_best['t_prop']          # along axis from axis origin
    L2_ax = g_best['t_exit']          # along axis from axis origin
    print(f"  L1 (t_prop, axis origin)     : {L1_ax*1e2:.2f} cm")
    print(f"  L2 (t_exit, axis origin)     : {L2_ax*1e2:.2f} cm")
    print(f"  L2 correct formula           : (R - w/2·cos α)/sin α = "
          f"{(R - w/2*np.cos(np.radians(best_alpha)))/np.sin(np.radians(best_alpha))*1e2:.2f} cm")
    print(f"  Verify: L1-3·L2+overlap+R_prop = "
          f"{(L1_ax - 3*L2_ax + overlap + R_prop)*1e2:.2f} cm  "
          f"({(L1_ax - 3*L2_ax + overlap + R_prop)*1e3:.2f} mm)")

# ── figure 1: top-view ────────────────────────────────────────────────────────

fig = plt.figure(figsize=(13, 9))
ax    = fig.add_axes([0.05, 0.08, 0.65, 0.88])
ax_xs = fig.add_axes([0.74, 0.30, 0.22, 0.45])
draw(R, L, w, best_alpha, R_prop, gap, overlap)

# ── figure 2: overhang sweep ──────────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2b = ax2.twinx()   # second y-axis for clearance

# shade aero-invalid region
ax2.fill_between(alphas, 0, 1, where=~aero_ok,
                 transform=ax2.get_xaxis_transform(),
                 color='orange', alpha=0.12, label=f'aero clearance < {gap_prop*1e3:.0f} mm (cross-pair)')

ax2.plot(alphas[valid], hangs[valid] * 1e3, color='steelblue', lw=2, label='overhang (prop tip)')
ax2.axhline(R_prop * 1e3, color='purple', ls='-.', lw=1.2,
            label=f'R_prop floor = {R_prop*1e3:.0f} mm')
ax2.axvline(best_alpha, color='red', ls='--', lw=1.5,
            label=f'best valid: {best_hang*1e3:.1f} mm  @  α={best_alpha:.1f}°')
ax2.fill_between(alphas[valid], hangs[valid]*1e3, R_prop*1e3,
                 where=(hangs[valid]*1e3 > R_prop*1e3), alpha=0.10, color='red')
ax2.fill_between(alphas[valid], hangs[valid]*1e3, R_prop*1e3,
                 where=(hangs[valid]*1e3 <= R_prop*1e3 + 0.01), alpha=0.10, color='green')

ax2b.plot(alphas[clr_valid], clrs[clr_valid] * 1e3, color='darkorange',
          lw=1.5, ls='--', label='adj prop clearance')
ax2b.axhline(gap_prop * 1e3, color='darkorange', ls=':', lw=1.0,
             label=f'gap_prop limit = {gap_prop*1e3:.0f} mm')
ax2b.set_ylabel('Adjacent prop clearance [mm]', color='darkorange', fontsize=11)
ax2b.tick_params(axis='y', labelcolor='darkorange')
ax2b.legend(loc='lower right', fontsize=9)

ax2.set_xlabel('α [deg]', fontsize=12)
ax2.set_ylabel('Prop tip overhang when retracted [mm]', fontsize=12)
ax2.set_title(
    f'Overhang vs α  |  R={R*1e2:.0f}cm  L={L*1e2:.0f}cm  w={w*1e3:.0f}mm  '
    f'R_prop={R_prop*1e2:.0f}cm  gap={gap*1e2:.0f}cm  gap_prop={gap_prop*1e2:.0f}cm  overlap={overlap*1e2:.0f}cm',
    fontsize=9)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()

plt.show()
