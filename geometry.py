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

# ==========================================================
# B. DERIVED GEOMETRY
# ==========================================================

# X-config arm angles (motor 0=FR, 1=FL, 2=RL, 3=RR)
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

    motor_labels = ['FR', 'FL', 'RL', 'RR']
    for i in range(4):
        start = np.array([0.0, 0.0, z_attach]) - CoM
        end   = motor_pos_c[i]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'k-', lw=3)
        ax.scatter(*motor_pos_c[i], s=120, color='orangered', zorder=5)
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
    ax.legend()

    plt.tight_layout()
    plt.show()
