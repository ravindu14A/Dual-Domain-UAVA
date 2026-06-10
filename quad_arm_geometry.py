"""
Telescoping X-quad arm optimiser.

Finds the arm angle alpha that minimises prop hangover when arms are fully
retracted.  Hangover = L - 3*L1 + overlap, capped at 0 (0 = fully flush).

Geometry
--------
Pill body: hemispheres of radius R + cylinder of length L_CYL.
Four arms (X-quad).  Each diagonal pair is offset +/-W from the diagonal
centre-line so their midpoints pass through the COM, letting them slide
past each other.  The two pairs live on different planes.

For one arm:
  axis direction : (cos a, sin a)
  lateral offset : W  (perpendicular, in top-view plane)
  origin         : (-W*sin a,  W*cos a)

  t_exit : arm exits pill on prop side
  t_entry: arm exits pill on retraction side (negative)
  L1     = t_exit - W*tan(a)   distance COM cross-section -> pill exit
  L      = L1 + R_prop + gap   distance COM cross-section -> prop centre
  hangover = L - 3*L1 + overlap  ~  arm_ext + overlap - chord
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETERS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────
R        = 0.10    # [m]  hemisphere radius (= body half-width)
L_CYL    = 0.55   # [m]  cylinder section length
W        = 0.04   # [m]  arm tube width  (each arm offset W from diagonal)
R_PROP   = 0.3048 # [m]  propeller radius
GAP      = 0.05   # [m]  prop-tip to body-surface clearance when extended
OVERLAP  = 0.0    # [m]  structural overlap of tube inside body
PROP_GAP = 0.05   # [m]  minimum tip-to-tip clearance between adjacent props

# ─────────────────────────────────────────────────────────────────────────────
#  PILL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def pill_outline(R, L, n=400):
    th_r = np.linspace(-np.pi / 2,  np.pi / 2,  n // 4)
    th_l = np.linspace( np.pi / 2, 3 * np.pi / 2, n // 4)
    xs = np.concatenate([
        L / 2 + R * np.cos(th_r),
        np.linspace(L / 2, -L / 2, n // 4),
        -L / 2 + R * np.cos(th_l),
        np.linspace(-L / 2, L / 2, n // 4),
    ])
    ys = np.concatenate([
        R * np.sin(th_r),
        np.full(n // 4,  R),
        R * np.sin(th_l),
        np.full(n // 4, -R),
    ])
    return xs, ys


def body_crossings(alpha, w_off, R, L):
    """Vectorised pill-arm intersection.  Returns (t_exit, t_entry)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ox, oy = -w_off * sa, w_off * ca
    search = 1.5 * (R + L / 2 + 1.0)
    ts = np.linspace(-search, search, 30000)
    xs = ox + ts * ca
    ys = oy + ts * sa

    inside = np.zeros(len(ts), dtype=bool)
    mid   = (xs >= -L / 2) & (xs <= L / 2)
    right = xs > L / 2
    left  = xs < -L / 2
    inside[mid]   = ys[mid] ** 2 <= R ** 2 + 1e-9
    inside[right] = (xs[right] - L / 2) ** 2 + ys[right] ** 2 <= R ** 2 + 1e-9
    inside[left]  = (xs[left]  + L / 2) ** 2 + ys[left]  ** 2 <= R ** 2 + 1e-9

    idx = np.where(np.diff(inside.astype(int)) != 0)[0]
    if len(idx) < 2:
        return None, None
    crossings = (ts[idx] + ts[idx + 1]) / 2
    pos = crossings[crossings > 0]
    neg = crossings[crossings < 0]
    if not len(pos) or not len(neg):
        return None, None
    return float(pos.min()), float(neg.max())


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def hangover_at(alpha_deg):
    alpha = np.radians(alpha_deg)
    t_exit, _ = body_crossings(alpha, W, R, L_CYL)
    if t_exit is None:
        return np.nan
    L1 = t_exit - W * np.tan(alpha)
    L  = L1 + R_PROP + GAP
    return L - 3 * L1 + OVERLAP   # uncapped; 0 = flush, >0 = sticks out


def min_prop_clearance(alpha_deg):
    """Minimum tip-to-tip clearance between any two adjacent props [m].

    By symmetry, adjacent-pair centre distances are:
      d1 = 2*|sin(a)| * sqrt(W^2 + t_prop^2)
      d2 = 2*|cos(a)| * sqrt(W^2 + t_prop^2)
    The smaller one drives the constraint.
    """
    alpha = np.radians(alpha_deg)
    t_exit, _ = body_crossings(alpha, W, R, L_CYL)
    if t_exit is None:
        return np.nan
    t_prop = t_exit + R_PROP + GAP
    r = np.sqrt(W ** 2 + t_prop ** 2)
    d_min = 2 * min(abs(np.sin(alpha)), abs(np.cos(alpha))) * r
    return d_min - 2 * R_PROP   # tip-to-tip


# ─────────────────────────────────────────────────────────────────────────────
#  SWEEP & OPTIMISE
# ─────────────────────────────────────────────────────────────────────────────

alphas     = np.linspace(5, 85, 800)
hangs_raw  = np.array([hangover_at(a)        for a in alphas])
clearances = np.array([min_prop_clearance(a) for a in alphas])
hangs_plot = np.maximum(hangs_raw, 0.0)   # cap at 0 for display

feasible = np.isfinite(hangs_raw) & (clearances >= PROP_GAP)
if feasible.any():
    best_i    = int(np.argmin(hangs_raw[feasible]))
    alpha_opt = alphas[feasible][best_i]
    hang_opt  = hangs_raw[feasible][best_i]
else:
    valid     = np.isfinite(hangs_raw)
    best_i    = int(np.argmin(hangs_raw[valid]))
    alpha_opt = alphas[valid][best_i]
    hang_opt  = hangs_raw[valid][best_i]
    print("WARNING: no feasible alpha -- prop-gap constraint cannot be satisfied.")

# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRY AT OPTIMAL ALPHA
# ─────────────────────────────────────────────────────────────────────────────

alpha_r = np.radians(alpha_opt)
t_exit, t_entry = body_crossings(alpha_r, W, R, L_CYL)
L1      = t_exit - W * np.tan(alpha_r)
L_arm   = L1 + R_PROP + GAP
chord   = t_exit - t_entry
arm_ext = R_PROP + GAP
L_tube  = arm_ext + OVERLAP
clr_opt = min_prop_clearance(alpha_opt)

print("=" * 52)
print(f"  Body:     R = {R*1e3:.0f} mm   L_cyl = {L_CYL*1e3:.0f} mm")
print(f"  Arm:      W = {W*1e3:.0f} mm   overlap = {OVERLAP*1e3:.0f} mm")
print(f"  Prop:     R_prop = {R_PROP*1e3:.0f} mm   gap = {GAP*1e3:.0f} mm")
print(f"  Prop gap required: {PROP_GAP*1e3:.0f} mm")
print("=" * 52)
print(f"  Optimal alpha      :  {alpha_opt:.1f} deg")
print(f"  L1 (COM->exit)     :  {L1*1e3:.1f} mm")
print(f"  L  (COM->prop)     :  {L_arm*1e3:.1f} mm")
print(f"  Chord              :  {chord*1e3:.1f} mm")
print(f"  Tube length        :  {L_tube*1e3:.1f} mm")
print(f"  Hangover           :  {max(hang_opt, 0)*1e3:.1f} mm  "
      f"({'flush' if hang_opt <= 0 else 'sticks out'})")
print(f"  Adj prop clearance :  {clr_opt*1e3:.1f} mm")
print("=" * 52)

# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_tube(ax, ox, oy, ca, sa, w, t0, t1, **kw):
    hw = w / 2
    px, py = -sa, ca
    c = np.array([
        [ox + t0 * ca + hw * px, oy + t0 * sa + hw * py],
        [ox + t0 * ca - hw * px, oy + t0 * sa - hw * py],
        [ox + t1 * ca - hw * px, oy + t1 * sa - hw * py],
        [ox + t1 * ca + hw * px, oy + t1 * sa + hw * py],
    ])
    ax.add_patch(plt.Polygon(c, **kw))


def draw_state(ax, alpha_deg, retracted=False):
    """Draw the vehicle top-view in either extended or retracted state."""
    alpha = np.radians(alpha_deg)
    xs, ys = pill_outline(R, L_CYL)
    ax.fill(xs, ys, color='#dce8f5', zorder=1)
    ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]),
            color='steelblue', lw=1.8, zorder=2)

    ARMS = [
        ( alpha,          W, '#e63946'),
        (-alpha,          W, '#2a9d8f'),
        (np.pi + alpha,   W, '#e63946'),
        (np.pi - alpha,   W, '#2a9d8f'),
    ]

    for (angle, w_off, col) in ARMS:
        te, tn = body_crossings(angle, w_off, R, L_CYL)
        if te is None:
            continue
        ca, sa = np.cos(angle), np.sin(angle)
        ox, oy = -w_off * sa, w_off * ca
        t_prop      = te + arm_ext
        t_ret_outer = tn + L_tube
        t_inner_ext = te - OVERLAP

        if retracted:
            draw_tube(ax, ox, oy, ca, sa, W, tn, t_ret_outer,
                      facecolor=col, edgecolor='none', alpha=0.75, zorder=4)
            px = ox + t_ret_outer * ca
            py = oy + t_ret_outer * sa
        else:
            draw_tube(ax, ox, oy, ca, sa, W, t_inner_ext, t_prop,
                      facecolor=col, edgecolor='none', alpha=0.75, zorder=4)
            px = ox + t_prop * ca
            py = oy + t_prop * sa

        # prop disc (filled transparent + outline)
        ax.add_patch(plt.Circle((px, py), R_PROP,
                                color=col, fill=True, alpha=0.18, lw=0, zorder=5))
        ax.add_patch(plt.Circle((px, py), R_PROP,
                                color=col, fill=False, lw=2.0, zorder=6))
        ax.plot(px, py, 'o', color=col, ms=4, zorder=7)

    lim = R + L_CYL / 2 + R_PROP + GAP + 0.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    if retracted:
        sub = f'hangover = {max(hang_opt, 0)*1e3:.1f} mm'
    else:
        sub = f'adj prop clearance = {clr_opt*1e3:.1f} mm'
    ax.set_title(f'{"Retracted" if retracted else "Extended"}  |  '
                 f'alpha = {alpha_deg:.1f} deg  |  {sub}', fontsize=11)

    leg = [
        mpatches.Patch(color='#e63946', alpha=0.75, label='Pair A'),
        mpatches.Patch(color='#2a9d8f', alpha=0.75, label='Pair B'),
    ]
    ax.legend(handles=leg, loc='upper right', fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1: extended vs retracted
# ─────────────────────────────────────────────────────────────────────────────

fig1, (ax_ext, ax_ret) = plt.subplots(1, 2, figsize=(14, 7))
draw_state(ax_ext, alpha_opt, retracted=False)
draw_state(ax_ret, alpha_opt, retracted=True)
fig1.suptitle(f'Optimal alpha = {alpha_opt:.1f} deg', fontsize=13)
plt.tight_layout()

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2: hangover curve with feasibility shading
# ─────────────────────────────────────────────────────────────────────────────

fig2, ax_hang = plt.subplots(figsize=(8, 5))

infeasible = clearances < PROP_GAP
ax_hang.fill_between(alphas, 0, hangs_plot * 1e3,
                     where=infeasible, color='#e63946', alpha=0.25,
                     label='prop clash (infeasible)')
ax_hang.plot(alphas, hangs_plot * 1e3, color='steelblue', lw=2, label='hangover')
ax_hang.fill_between(alphas, 0, hangs_plot * 1e3,
                     where=~infeasible, color='steelblue', alpha=0.12)
ax_hang.axvline(alpha_opt, color='red', lw=1.5, ls='--',
                label=f'optimal  alpha = {alpha_opt:.1f} deg')
ax_hang.axhline(0, color='k', lw=1.0)
ax_hang.set_xlabel('alpha  [deg]')
ax_hang.set_ylabel('Hangover  [mm]')
ax_hang.set_title('Hangover vs arm angle  (capped at 0, red shading = props clash)')
ax_hang.legend(fontsize=9)
ax_hang.grid(True, alpha=0.3)
ax_hang.set_ylim(bottom=0)

plt.tight_layout()
plt.show()
