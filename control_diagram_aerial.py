"""
Aerial controller block diagram — UAUV inspection mission.
Run standalone: python control_diagram_aerial.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#0d0d0d')
ax.set_facecolor('#0d0d0d')

# ── colour palette ──────────────────────────────────────────────
C_BOX   = '#1a1a2e'
C_EDGE  = '#ffffff'
C_TEXT  = '#ffffff'
C_REF   = '#7ec8e3'    # light blue  — reference signals
C_ERR   = '#f4a261'    # orange      — error signals
C_INNER = '#a8d8a8'    # green       — inner-loop outputs
C_MIX   = '#c77dff'    # purple      — mixing / motor
C_STATE = '#ffd166'    # yellow      — state feedback
C_SUMJ  = '#ffffff'

def box(cx, cy, w, h, label, sublabel=None, color=C_EDGE, fontsize=9, labelsize=8):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=C_BOX, edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    y_text = cy + (0.15 if sublabel else 0)
    ax.text(cx, y_text, label, ha='center', va='center',
            color=color, fontsize=fontsize, fontweight='bold')
    if sublabel:
        ax.text(cx, cy - 0.28, sublabel, ha='center', va='center',
                color=color, fontsize=labelsize - 1, style='italic')

def sumjunc(cx, cy, r=0.22):
    circ = plt.Circle((cx, cy), r, facecolor=C_BOX, edgecolor=C_SUMJ, linewidth=1.5)
    ax.add_patch(circ)
    ax.text(cx, cy, '＋', ha='center', va='center', color=C_SUMJ, fontsize=9)

def arrow(x1, y1, x2, y2, color='white', lw=1.4, style='->', label=None, lx=None, ly=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))
    if label:
        ax.text(lx if lx else (x1+x2)/2,
                ly if ly else (y1+y2)/2 + 0.18,
                label, ha='center', va='bottom', color=color, fontsize=7.5)

def line(x1, y1, x2, y2, color='white', lw=1.2):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw)

# ════════════════════════════════════════════════════════════════
# ROW HEIGHTS
# ════════════════════════════════════════════════════════════════
Y_TOP  = 7.8   # reference inputs row
Y_MID  = 5.5   # main signal flow
Y_BOT  = 3.2   # state feedback row
Y_MOT  = 1.5   # motor / plant row

# ════════════════════════════════════════════════════════════════
# REFERENCE INPUTS
# ════════════════════════════════════════════════════════════════
ax.text(0.5, Y_TOP, 'Cylindrical\nreference\n(r, θ, z)\n(ṙ, θ̇, ż)',
        ha='center', va='center', color=C_REF, fontsize=8, fontweight='bold')

ax.text(0.5, Y_MOT + 1.0, 'Yaw ref\nψ_d = θ_ref + π\n(face tower)',
        ha='center', va='center', color=C_REF, fontsize=7.5, fontweight='bold')

# ════════════════════════════════════════════════════════════════
# SUM JUNCTIONS — position errors
# ════════════════════════════════════════════════════════════════
SJ1x = 2.0    # r / θ error junction
SJ2x = 2.0    # z error junction  (below)

sumjunc(SJ1x, Y_TOP - 0.5)
sumjunc(SJ1x, Y_BOT + 0.8)

# ════════════════════════════════════════════════════════════════
# OUTER LOOP — CYLINDRICAL POSITION PID
# ════════════════════════════════════════════════════════════════
box(4.0, Y_TOP - 0.5, 2.0, 1.0,
    'Cyl. Position PID',
    'r, θ  →  aᵣ, aₜ',
    color=C_REF, fontsize=8)

box(4.0, Y_BOT + 0.8, 2.0, 0.9,
    'Altitude PID',
    'z  →  a_z',
    color=C_REF, fontsize=8)

# ════════════════════════════════════════════════════════════════
# CYL → CARTESIAN CONVERSION + ATTITUDE COMMAND
# ════════════════════════════════════════════════════════════════
box(6.9, Y_MID, 2.2, 2.2,
    'Cyl → Inertial\n+ Attitude cmd',
    'φ_d = −aᵧ / g\nθ_d =  aₓ / g\nF_z = m(g+a_z)',
    color=C_ERR, fontsize=8)

# ════════════════════════════════════════════════════════════════
# SUM JUNCTIONS — attitude errors
# ════════════════════════════════════════════════════════════════
SJ_phi   = (9.5, 7.2)
SJ_theta = (9.5, 5.5)
SJ_psi   = (9.5, 3.8)
SJ_thr   = (9.5, 2.2)

for sj in [SJ_phi, SJ_theta, SJ_psi, SJ_thr]:
    sumjunc(*sj)

# ════════════════════════════════════════════════════════════════
# INNER LOOP — ATTITUDE PIDs
# ════════════════════════════════════════════════════════════════
box(11.2, 7.2, 1.6, 0.8, 'Roll PID',   color=C_INNER, fontsize=8)
box(11.2, 5.5, 1.6, 0.8, 'Pitch PID',  color=C_INNER, fontsize=8)
box(11.2, 3.8, 1.6, 0.8, 'Yaw PID',   color=C_INNER, fontsize=8)
box(11.2, 2.2, 1.6, 0.8, 'Thrust',    color=C_INNER, fontsize=8,
    sublabel='F_z → T_cmd')

# ════════════════════════════════════════════════════════════════
# MOTOR MIXING
# ════════════════════════════════════════════════════════════════
box(13.5, 4.7, 1.8, 3.2,
    'Motor\nMixing\nA⁻¹',
    '[τφ,τθ,τψ,Fz]\n→[ω₁²..ω₄²]',
    color=C_MIX, fontsize=8)

# ════════════════════════════════════════════════════════════════
# PLANT
# ════════════════════════════════════════════════════════════════
box(16.0, 4.7, 2.0, 3.6,
    'Plant\n(ODE)',
    'Motor lag\n+ 6-DOF\nrigid body',
    color=C_STATE, fontsize=8)

# ════════════════════════════════════════════════════════════════
# STATE FEEDBACK BLOCK
# ════════════════════════════════════════════════════════════════
box(11.5, Y_BOT - 1.1, 3.5, 0.9,
    'State extraction',
    '[x,y,z,φ,θ,ψ,ẋ,ẏ,ż,p,q,r] → (r,θ,z,ṙ,θ̇,ż)',
    color=C_STATE, fontsize=7.5)

# ════════════════════════════════════════════════════════════════
# ARROWS — reference to outer PIDs
# ════════════════════════════════════════════════════════════════
arrow(0.9, Y_TOP, SJ1x - 0.22, Y_TOP - 0.5, color=C_REF, label='r_ref, θ_ref', lx=1.5, ly=Y_TOP + 0.1)
arrow(0.9, Y_BOT + 1.1, SJ1x - 0.22, Y_BOT + 0.8, color=C_REF, label='z_ref', lx=1.4, ly=Y_BOT + 1.25)

# outer PID → attitude cmd block
arrow(5.0, Y_TOP - 0.5, 5.9, Y_MID + 0.55, color=C_REF, label='aᵣ, aₜ', lx=5.5, ly=Y_MID + 1.1)
arrow(5.0, Y_BOT + 0.8, 5.9, Y_MID - 0.55, color=C_REF, label='a_z', lx=5.5, ly=Y_BOT + 0.95)

# attitude cmd → sum junctions
arrow(8.0, 6.9, SJ_phi[0] - 0.22,   SJ_phi[1],   color=C_ERR, label='φ_d', lx=8.9, ly=7.1)
arrow(8.0, 5.5, SJ_theta[0] - 0.22, SJ_theta[1], color=C_ERR, label='θ_d', lx=8.9, ly=5.65)
arrow(8.0, 2.5, SJ_psi[0] - 0.22,   SJ_psi[1],   color=C_ERR, label='ψ_d', lx=8.9, ly=2.7)
arrow(8.0, 4.2, SJ_thr[0] - 0.22,   SJ_thr[1],   color=C_ERR, label='F_z cmd', lx=8.8, ly=4.35)

# yaw ref arrow
arrow(0.9, Y_MOT + 0.95, 7.6, 2.4,  color=C_REF)

# sum junction → inner PIDs
arrow(SJ_phi[0] + 0.22,   SJ_phi[1],   10.4, 7.2,  color=C_ERR)
arrow(SJ_theta[0] + 0.22, SJ_theta[1], 10.4, 5.5,  color=C_ERR)
arrow(SJ_psi[0] + 0.22,   SJ_psi[1],  10.4, 3.8,  color=C_ERR)
arrow(SJ_thr[0] + 0.22,   SJ_thr[1],  10.4, 2.2,  color=C_ERR)

# inner PIDs → mixing
arrow(12.0, 7.2, 12.6, 6.0, color=C_INNER, label='τ_φ', lx=12.4, ly=6.8)
arrow(12.0, 5.5, 12.6, 5.3, color=C_INNER, label='τ_θ', lx=12.45, ly=5.65)
arrow(12.0, 3.8, 12.6, 4.1, color=C_INNER, label='τ_ψ', lx=12.4, ly=4.15)
arrow(12.0, 2.2, 12.6, 3.5, color=C_INNER, label='F_z', lx=12.4, ly=3.0)

# mixing → plant
arrow(14.4, 4.7, 15.0, 4.7, color=C_MIX, label='ω₁..ω₄', lx=14.7, ly=4.85)

# plant output → right side, wrap around bottom for feedback
arrow(17.0, 4.7, 17.8, 4.7, color=C_STATE)
ax.text(17.5, 4.85, 'full\nstate', ha='center', va='bottom', color=C_STATE, fontsize=7)

# feedback: plant bottom → state extraction
line(17.0, 3.0, 17.0, 1.0,  color=C_STATE)
line(17.0, 1.0, 11.5, 1.0,  color=C_STATE)
arrow(11.5, 1.0, 11.5, Y_BOT - 1.55, color=C_STATE, label='state feedback')

# state extraction → sum junctions (attitude)
line(11.5, Y_BOT - 1.1 - 0.45, 9.5, Y_BOT - 1.55, color=C_STATE)
line(9.5,  Y_BOT - 1.55, 9.5, 2.2 - 0.22,          color=C_STATE)

# also feed r,θ,z back to outer sum junctions
line(9.5, Y_BOT - 1.55, 2.0, Y_BOT - 1.55, color=C_STATE)
arrow(2.0, Y_BOT - 1.55, 2.0, Y_BOT + 0.8 - 0.22, color=C_STATE, label='r,θ,z\nṙ,θ̇,ż', lx=1.5, ly=Y_BOT - 0.7)
line(2.0, Y_BOT - 1.55, 2.0, Y_TOP - 0.5 - 0.22, color=C_STATE)

# attitude feedback to inner sum junctions
arrow(9.5, 2.22, 9.5, 2.2 - 0.22, color=C_STATE)
line(9.5, 3.8 - 0.22, 9.5, 2.22,  color=C_STATE)
line(9.5, 5.5 - 0.22, 9.5, 3.8,   color=C_STATE)
line(9.5, 7.2 - 0.22, 9.5, 5.5,   color=C_STATE)

# ════════════════════════════════════════════════════════════════
# LABELS — loop brackets
# ════════════════════════════════════════════════════════════════
ax.text(4.5, 9.5, 'OUTER LOOP  (cylindrical position PID)',
        ha='center', va='center', color='#7ec8e3', fontsize=10, fontweight='bold')
ax.text(11.2, 9.5, 'INNER LOOP  (attitude PID)',
        ha='center', va='center', color='#a8d8a8', fontsize=10, fontweight='bold')
ax.text(13.5, 9.5, 'MIXING',
        ha='center', va='center', color='#c77dff', fontsize=10, fontweight='bold')
ax.text(16.0, 9.5, 'PLANT',
        ha='center', va='center', color='#ffd166', fontsize=10, fontweight='bold')

# divider lines
for xv, col in [(8.15, '#555'), (12.35, '#555'), (14.7, '#555')]:
    line(xv, 1.2, xv, 9.2, color=col, lw=0.8)

ax.set_title('Aerial Controller — Cascaded PID Block Diagram',
             color='white', fontsize=13, fontweight='bold', pad=12)

plt.tight_layout()
plt.savefig('control_diagram_aerial.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved: control_diagram_aerial.png")
plt.show()
