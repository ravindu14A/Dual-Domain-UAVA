# ==========================================
# UAUV PRELIMINARY GEOMETRY & MASS MODEL
# ==========================================
# Central box + 4 arms (X-config) + 4 motor point masses.
# All dimensions in metres, masses in kg.
# Body frame: x forward, y left, z up.  Origin at vehicle CoM.
#
# FILL IN YOUR VALUES in Section A before running.
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==========================================================
# A. PARAMETERS  ← edit these
# ==========================================================

# Central box (solid uniform rectangular box)
L_box = 2   # [m] length  (x-axis)
W_box = 1   # [m] width   (y-axis)
H_box = 0.6   # [m] height  (z-axis)
m_box = 22.4    # [kg]

# Arms: 4 thin rods, X-config (45° offsets), emanating from top-centre of box
L_arm = 2   # [m] arm length (from box centre to motor)
m_arm = 0.15   # [kg] mass of each arm (uniform rod)

# Motors / rotors: point mass at tip of each arm
m_motor = 0.5  # [kg] per motor

# Underwater arm fold fraction: fold joint at this fraction of L_arm from base.
# After 180-degree fold, prop centre is at L_fold = f*L_arm - (1-f)*L_arm = (2f-1)*L_arm
# Must be > 0.5 for prop to remain outside box centre. Example: f=0.75 -> L_fold=0.5*L_arm
UW_ARM_FOLD_FRAC = 0.75

# ==========================================================
# B. DERIVED GEOMETRY
# ==========================================================

# X-config arm angles (body x=forward, y=left)
# motor 0=FL (45°), 1=RL (135°), 2=RR (225°), 3=FR (315°)
arm_angles_deg = [45.0, 135.0, 225.0, 315.0]
arm_angles_rad = np.radians(arm_angles_deg)

# Arms attach at the top face of the box (z = +H_box/2 in box frame)
z_attach = H_box / 2.0

# Arm CoM positions (midpoint of rod, horizontal)
arm_com = np.array([
    [(L_arm / 2) * np.cos(a), (L_arm / 2) * np.sin(a), z_attach]
    for a in arm_angles_rad
])

# Motor positions (tip of arm)
motor_pos = np.array([
    [L_arm * np.cos(a), L_arm * np.sin(a), z_attach]
    for a in arm_angles_rad
])

# ==========================================================
# C. TOTAL MASS & CENTRE OF MASS
# ==========================================================

total_mass = m_box + 4 * m_arm + 4 * m_motor

com_x = (m_box * 0.0 + m_arm * arm_com[:, 0].sum() + m_motor * motor_pos[:, 0].sum()) / total_mass
com_y = (m_box * 0.0 + m_arm * arm_com[:, 1].sum() + m_motor * motor_pos[:, 1].sum()) / total_mass
com_z = (m_box * 0.0 + m_arm * arm_com[:, 2].sum() + m_motor * motor_pos[:, 2].sum()) / total_mass
CoM = np.array([com_x, com_y, com_z])

# Shift all positions to CoM frame
arm_com_c   = arm_com   - CoM
motor_pos_c = motor_pos - CoM
box_com_c   = np.array([0.0, 0.0, 0.0]) - CoM

# ==========================================================
# D. MOMENTS OF INERTIA  (parallel axis theorem, about CoM)
# ==========================================================

# Box: solid cuboid
Ixx_box = m_box / 12.0 * (W_box**2 + H_box**2) + m_box * (box_com_c[1]**2 + box_com_c[2]**2)
Iyy_box = m_box / 12.0 * (L_box**2 + H_box**2) + m_box * (box_com_c[0]**2 + box_com_c[2]**2)
Izz_box = m_box / 12.0 * (L_box**2 + W_box**2) + m_box * (box_com_c[0]**2 + box_com_c[1]**2)

# Arms: uniform thin rods at angle a, elevated to z_attach
Ixx_arms = Iyy_arms = Izz_arms = 0.0
for i, a in enumerate(arm_angles_rad):
    cx, cy, cz = arm_com_c[i]
    ux, uy = np.cos(a), np.sin(a)
    # Rod inertia about its own CoM: transverse axes only (thin rod)
    I_rod_x = m_arm * L_arm**2 / 12.0 * (uy**2)
    I_rod_y = m_arm * L_arm**2 / 12.0 * (ux**2)
    I_rod_z = m_arm * L_arm**2 / 12.0
    Ixx_arms += I_rod_x + m_arm * (cy**2 + cz**2)
    Iyy_arms += I_rod_y + m_arm * (cx**2 + cz**2)
    Izz_arms += I_rod_z + m_arm * (cx**2 + cy**2)

# Motors: point masses
Ixx_motors = sum(m_motor * (r[1]**2 + r[2]**2) for r in motor_pos_c)
Iyy_motors = sum(m_motor * (r[0]**2 + r[2]**2) for r in motor_pos_c)
Izz_motors = sum(m_motor * (r[0]**2 + r[1]**2) for r in motor_pos_c)

Ixx = Ixx_box + Ixx_arms + Ixx_motors
Iyy = Iyy_box + Iyy_arms + Iyy_motors
Izz = Izz_box + Izz_arms + Izz_motors

# ==========================================================
# D_UW. UNDERWATER INERTIA  (folded-arm configuration)
# ==========================================================
# When arms fold at UW_ARM_FOLD_FRAC, each arm splits into two segments:
#   Seg1: box → fold joint,  length L1 = frac*L_arm,      mass m1 = m_arm*frac
#   Seg2: fold joint → motor, length L2 = (1-frac)*L_arm,  mass m2 = m_arm*(1-frac)
#         (Seg2 runs in opposite direction back toward centre)
# Motor lands at L_fold = (2*frac-1)*L_arm from centre, same z as aerial.
# Total mass and CoM are unchanged (all z positions still at H_box/2).

_uw_frac = UW_ARM_FOLD_FRAC
_uw_L1   = _uw_frac * L_arm
_uw_L2   = (1.0 - _uw_frac) * L_arm
_uw_m1   = m_arm * _uw_frac
_uw_m2   = m_arm * (1.0 - _uw_frac)
UW_L_fold = (2.0 * _uw_frac - 1.0) * L_arm

# UW motor positions (horizontal extent changed, z same)
uw_motor_pos = np.array([
    [UW_L_fold * np.cos(a), UW_L_fold * np.sin(a), z_attach]
    for a in arm_angles_rad
])
uw_motor_pos_c = uw_motor_pos - CoM   # CoM is unchanged

# Segment CoM positions (in absolute frame, then shift to CoM frame)
uw_seg1_com = np.array([
    [(_uw_frac / 2.0) * L_arm * np.cos(a),
     (_uw_frac / 2.0) * L_arm * np.sin(a), z_attach]
    for a in arm_angles_rad
])
uw_seg2_com = np.array([
    [((3.0 * _uw_frac - 1.0) / 2.0) * L_arm * np.cos(a),
     ((3.0 * _uw_frac - 1.0) / 2.0) * L_arm * np.sin(a), z_attach]
    for a in arm_angles_rad
])
uw_seg1_com_c = uw_seg1_com - CoM
uw_seg2_com_c = uw_seg2_com - CoM

# Arm inertia: two segments per arm (sin²/cos² same for opposite rod direction)
Ixx_arms_uw = Iyy_arms_uw = Izz_arms_uw = 0.0
for i, a in enumerate(arm_angles_rad):
    ux, uy = np.cos(a), np.sin(a)
    for m_seg, L_seg, (cx, cy, cz) in [
        (_uw_m1, _uw_L1, uw_seg1_com_c[i]),
        (_uw_m2, _uw_L2, uw_seg2_com_c[i]),
    ]:
        I_x = m_seg * L_seg**2 / 12.0 * (uy**2)
        I_y = m_seg * L_seg**2 / 12.0 * (ux**2)
        I_z = m_seg * L_seg**2 / 12.0
        Ixx_arms_uw += I_x + m_seg * (cy**2 + cz**2)
        Iyy_arms_uw += I_y + m_seg * (cx**2 + cz**2)
        Izz_arms_uw += I_z + m_seg * (cx**2 + cy**2)

# Motors UW: point masses at folded positions
Ixx_motors_uw = sum(m_motor * (r[1]**2 + r[2]**2) for r in uw_motor_pos_c)
Iyy_motors_uw = sum(m_motor * (r[0]**2 + r[2]**2) for r in uw_motor_pos_c)
Izz_motors_uw = sum(m_motor * (r[0]**2 + r[1]**2) for r in uw_motor_pos_c)

Ixx_uw = Ixx_box + Ixx_arms_uw + Ixx_motors_uw
Iyy_uw = Iyy_box + Iyy_arms_uw + Iyy_motors_uw
Izz_uw = Izz_box + Izz_arms_uw + Izz_motors_uw

# ==========================================================
# E. PRINT RESULTS  &  F. 3D VISUALISATION  (only when run directly)
# ==========================================================
if __name__ == "__main__":
    print("=" * 50)
    print("UAUV PRELIMINARY MASS PROPERTIES")
    print("=" * 50)
    print(f"  Box             : {m_box:.3f} kg   ({L_box:.3f} x {W_box:.3f} x {H_box:.3f} m)")
    print(f"  Arms (x4)       : {4*m_arm:.3f} kg   ({m_arm:.3f} kg each, L={L_arm:.3f} m)")
    print(f"  Motors (x4)     : {4*m_motor:.3f} kg   ({m_motor:.3f} kg each)")
    print(f"  TOTAL MASS      : {total_mass:.3f} kg")
    print()
    print(f"  Centre of Mass  : [{CoM[0]:+.4f},  {CoM[1]:+.4f},  {CoM[2]:+.4f}] m")
    print(f"  (positive z_CoM = CoM is above box geometric centre)")
    print()
    print(f"  Ixx (roll)      : {Ixx:.5f} kg·m²")
    print(f"  Iyy (pitch)     : {Iyy:.5f} kg·m²")
    print(f"  Izz (yaw)       : {Izz:.5f} kg·m²")
    print("=" * 50)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("UAUV Preliminary Geometry", fontsize=13, fontweight='bold')

    info = (f"Total mass : {total_mass:.2f} kg\n"
            f"Ixx : {Ixx:.4f} kg·m²\n"
            f"Iyy : {Iyy:.4f} kg·m²\n"
            f"Izz : {Izz:.4f} kg·m²")
    fig.text(0.02, 0.97, info, fontsize=9, va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    def draw_box(ax, cx, cy, cz, lx, ly, lz, color='steelblue', alpha=0.35):
        x0, x1 = cx - lx/2, cx + lx/2
        y0, y1 = cy - ly/2, cy + ly/2
        z0, z1 = cz - lz/2, cz + lz/2
        verts = [
            [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
            [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
            [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
            [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
            [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],
            [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],
        ]
        poly = Poly3DCollection(verts, alpha=alpha, edgecolor='k', linewidth=0.5)
        poly.set_facecolor(color)
        ax.add_collection3d(poly)

    draw_box(ax, box_com_c[0], box_com_c[1], box_com_c[2], L_box, W_box, H_box,
             color='steelblue', alpha=0.45)

    motor_labels = ['FL', 'RL', 'RR', 'FR']
    spin_color   = ['tomato', 'royalblue', 'tomato', 'royalblue']  # red=CCW, blue=CW — diagonal pairs
    prop_r       = L_arm * 0.12   # propeller disk radius

    _ang = np.linspace(0, 2 * np.pi, 40)
    for i in range(4):
        start = np.array([0.0, 0.0, z_attach]) - CoM
        end   = motor_pos_c[i]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'k-', lw=3)

        cx, cy, cz = motor_pos_c[i]
        xd = cx + prop_r * np.cos(_ang)
        yd = cy + prop_r * np.sin(_ang)
        zd = np.full(40, cz)
        disk = Poly3DCollection([list(zip(xd, yd, zd))],
                                alpha=0.75, facecolor=spin_color[i],
                                edgecolor='k', linewidth=0.8)
        ax.add_collection3d(disk)

        ax.text(motor_pos_c[i, 0]*1.12, motor_pos_c[i, 1]*1.12, motor_pos_c[i, 2],
                motor_labels[i], fontsize=9, ha='center')

    ax.scatter(0, 0, 0, s=200, color='gold', marker='*', zorder=10, label='CoM')

    axis_len = L_arm * 0.4
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='r', arrow_length_ratio=0.2)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='g', arrow_length_ratio=0.2)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='b', arrow_length_ratio=0.2)
    ax.text(axis_len*1.15, 0, 0, 'x', color='r', fontsize=10)
    ax.text(0, axis_len*1.15, 0, 'y', color='g', fontsize=10)
    ax.text(0, 0, axis_len*1.15, 'z', color='b', fontsize=10)

    span = L_arm * 1.4
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(-span, span)
    ax.set_xlabel('x  [m]')
    ax.set_ylabel('y  [m]')
    ax.set_zlabel('z  [m]')
    ax.set_box_aspect([1, 1, 1])

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        ax.get_legend_handles_labels()[0][0],   # CoM star
        mpatches.Patch(facecolor='tomato',     edgecolor='k', label='CCW rotor'),
        mpatches.Patch(facecolor='royalblue',  edgecolor='k', label='CW rotor'),
    ], labels=['CoM', 'CCW rotor', 'CW rotor'])

    plt.tight_layout()

    # ── Figure 2: Underwater (folded-arm) configuration ───────────────────────
    fig2 = plt.figure(figsize=(9, 8))
    ax2  = fig2.add_subplot(111, projection='3d')
    ax2.set_title("UAUV Underwater Geometry  (arms folded)", fontsize=13, fontweight='bold')

    info_uw = (f"Total mass : {total_mass:.2f} kg  (unchanged)\n"
               f"L_fold     : {UW_L_fold:.3f} m  (frac={UW_ARM_FOLD_FRAC})\n"
               f"Ixx_uw : {Ixx_uw:.4f} kg·m²\n"
               f"Iyy_uw : {Iyy_uw:.4f} kg·m²\n"
               f"Izz_uw : {Izz_uw:.4f} kg·m²")
    fig2.text(0.02, 0.97, info_uw, fontsize=9, va='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # Same box
    draw_box(ax2, box_com_c[0], box_com_c[1], box_com_c[2], L_box, W_box, H_box,
             color='steelblue', alpha=0.45)

    prop_r_uw  = UW_L_fold * 0.18   # prop disk radius scaled to folded arm length

    for i in range(4):
        a     = arm_angles_rad[i]
        fold  = np.array([UW_ARM_FOLD_FRAC * L_arm * np.cos(a),
                          UW_ARM_FOLD_FRAC * L_arm * np.sin(a),
                          z_attach]) - CoM
        start = np.array([0.0, 0.0, z_attach]) - CoM
        motor = uw_motor_pos_c[i]

        # Segment 1: box centre → fold joint
        ax2.plot([start[0], fold[0]], [start[1], fold[1]], [start[2], fold[2]],
                 'k-', lw=3)
        # Segment 2: fold joint → motor (lighter line, folded-back portion)
        ax2.plot([fold[0], motor[0]], [fold[1], motor[1]], [fold[2], motor[2]],
                 color='gray', lw=2, ls='--')
        # Fold joint marker
        ax2.scatter(*fold, s=60, color='black', zorder=6)

        # Prop disk at motor position
        cx, cy, cz = motor
        xd = cx + prop_r_uw * np.cos(_ang)
        yd = cy + prop_r_uw * np.sin(_ang)
        zd = np.full(40, cz)
        disk2 = Poly3DCollection([list(zip(xd, yd, zd))],
                                 alpha=0.75, facecolor=spin_color[i],
                                 edgecolor='k', linewidth=0.8)
        ax2.add_collection3d(disk2)

        ax2.text(motor[0]*1.18, motor[1]*1.18, motor[2],
                 motor_labels[i], fontsize=9, ha='center')

    ax2.scatter(0, 0, 0, s=200, color='gold', marker='*', zorder=10, label='CoM')

    axis_len2 = UW_L_fold * 0.6
    ax2.quiver(0, 0, 0, axis_len2, 0, 0, color='r', arrow_length_ratio=0.2)
    ax2.quiver(0, 0, 0, 0, axis_len2, 0, color='g', arrow_length_ratio=0.2)
    ax2.quiver(0, 0, 0, 0, 0, axis_len2, color='b', arrow_length_ratio=0.2)
    ax2.text(axis_len2*1.15, 0, 0, 'x', color='r', fontsize=10)
    ax2.text(0, axis_len2*1.15, 0, 'y', color='g', fontsize=10)
    ax2.text(0, 0, axis_len2*1.15, 'z', color='b', fontsize=10)

    span2 = max(L_box, W_box) * 0.9
    ax2.set_xlim(-span2, span2)
    ax2.set_ylim(-span2, span2)
    ax2.set_zlim(-span2, span2)
    ax2.set_xlabel('x  [m]')
    ax2.set_ylabel('y  [m]')
    ax2.set_zlabel('z  [m]')
    ax2.set_box_aspect([1, 1, 1])

    ax2.legend(handles=[
        ax2.get_legend_handles_labels()[0][0],
        mpatches.Patch(facecolor='tomato',    edgecolor='k', label='CCW rotor'),
        mpatches.Patch(facecolor='royalblue', edgecolor='k', label='CW rotor'),
        plt.Line2D([0],[0], color='k',    lw=3,          label='Arm seg 1'),
        plt.Line2D([0],[0], color='gray', lw=2, ls='--', label='Arm seg 2 (folded)'),
    ], labels=['CoM', 'CCW rotor', 'CW rotor', 'Arm seg 1', 'Arm seg 2 (folded)'])

    fig2.tight_layout()
    plt.show()
