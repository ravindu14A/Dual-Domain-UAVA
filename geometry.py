# UAUV geometry and mass model
# central box + 4 arms (X-config) + 4 motor point masses
# dimensions in metres, masses in kg
# body frame: x forward, y left, z up, origin at CoM

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- parameters (edit these) ---

# central box
L_box = 0.8   # [m] x
W_box = 0.4   # [m] y
H_box = 0.4   # [m] z
m_box = 22.4  # [kg]

# 4 arms in X-config, 45 deg offsets from box top-centre
L_arm = 1      # [m]
m_arm = 0.15   # [kg] each

m_motor = 0.5  # [kg] per motor (point mass at arm tip)

# underwater fold fraction: fold joint at frac*L_arm from base
# after 180 deg fold: L_fold = (2f-1)*L_arm  (must be > 0.5)
UW_ARM_FOLD_FRAC = 0.75

# --- derived geometry ---

# arm angles: FL=45, RL=135, RR=225, FR=315
arm_angles_deg = [45.0, 135.0, 225.0, 315.0]
arm_angles_rad = np.radians(arm_angles_deg)

z_attach = H_box / 2.0   # arms attach at box top face

arm_com = np.array([
    [(L_arm / 2) * np.cos(a), (L_arm / 2) * np.sin(a), z_attach]
    for a in arm_angles_rad
])

motor_pos = np.array([
    [L_arm * np.cos(a), L_arm * np.sin(a), z_attach]
    for a in arm_angles_rad
])

# --- total mass and CoM ---

total_mass = m_box + 4 * m_arm + 4 * m_motor

com_x = (m_box * 0.0 + m_arm * arm_com[:, 0].sum() + m_motor * motor_pos[:, 0].sum()) / total_mass
com_y = (m_box * 0.0 + m_arm * arm_com[:, 1].sum() + m_motor * motor_pos[:, 1].sum()) / total_mass
com_z = (m_box * 0.0 + m_arm * arm_com[:, 2].sum() + m_motor * motor_pos[:, 2].sum()) / total_mass
CoM = np.array([com_x, com_y, com_z])

# shift to CoM frame
arm_com_c   = arm_com   - CoM
motor_pos_c = motor_pos - CoM
box_com_c   = np.array([0.0, 0.0, 0.0]) - CoM

# --- moments of inertia (parallel axis theorem, about CoM) ---

Ixx_box = m_box / 12.0 * (W_box**2 + H_box**2) + m_box * (box_com_c[1]**2 + box_com_c[2]**2)
Iyy_box = m_box / 12.0 * (L_box**2 + H_box**2) + m_box * (box_com_c[0]**2 + box_com_c[2]**2)
Izz_box = m_box / 12.0 * (L_box**2 + W_box**2) + m_box * (box_com_c[0]**2 + box_com_c[1]**2)

# arms: thin rods at angle a
Ixx_arms = Iyy_arms = Izz_arms = 0.0
for i, a in enumerate(arm_angles_rad):
    cx, cy, cz = arm_com_c[i]
    ux, uy = np.cos(a), np.sin(a)
    I_rod_x = m_arm * L_arm**2 / 12.0 * (uy**2)
    I_rod_y = m_arm * L_arm**2 / 12.0 * (ux**2)
    I_rod_z = m_arm * L_arm**2 / 12.0
    Ixx_arms += I_rod_x + m_arm * (cy**2 + cz**2)
    Iyy_arms += I_rod_y + m_arm * (cx**2 + cz**2)
    Izz_arms += I_rod_z + m_arm * (cx**2 + cy**2)

Ixx_motors = sum(m_motor * (r[1]**2 + r[2]**2) for r in motor_pos_c)
Iyy_motors = sum(m_motor * (r[0]**2 + r[2]**2) for r in motor_pos_c)
Izz_motors = sum(m_motor * (r[0]**2 + r[1]**2) for r in motor_pos_c)

Ixx = Ixx_box + Ixx_arms + Ixx_motors
Iyy = Iyy_box + Iyy_arms + Iyy_motors
Izz = Izz_box + Izz_arms + Izz_motors

# --- underwater inertia (folded-arm config) ---
# each arm splits at UW_ARM_FOLD_FRAC into two segments running in opposite directions
# motor ends up at L_fold = (2*frac-1)*L_arm; total mass and CoM unchanged

_uw_frac = UW_ARM_FOLD_FRAC
_uw_L1   = _uw_frac * L_arm
_uw_L2   = (1.0 - _uw_frac) * L_arm
_uw_m1   = m_arm * _uw_frac
_uw_m2   = m_arm * (1.0 - _uw_frac)
UW_L_fold = (2.0 * _uw_frac - 1.0) * L_arm

uw_motor_pos = np.array([
    [UW_L_fold * np.cos(a), UW_L_fold * np.sin(a), z_attach]
    for a in arm_angles_rad
])
uw_motor_pos_c = uw_motor_pos - CoM

# segment CoM positions
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

# arm inertia: two segments per arm
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

# motor inertia at folded positions
Ixx_motors_uw = sum(m_motor * (r[1]**2 + r[2]**2) for r in uw_motor_pos_c)
Iyy_motors_uw = sum(m_motor * (r[0]**2 + r[2]**2) for r in uw_motor_pos_c)
Izz_motors_uw = sum(m_motor * (r[0]**2 + r[1]**2) for r in uw_motor_pos_c)

Ixx_uw = Ixx_box + Ixx_arms_uw + Ixx_motors_uw
Iyy_uw = Iyy_box + Iyy_arms_uw + Iyy_motors_uw
Izz_uw = Izz_box + Izz_arms_uw + Izz_motors_uw

# run directly for printed results and 3D visualisation
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
    spin_color   = ['tomato', 'royalblue', 'tomato', 'royalblue']  # red=CCW, blue=CW
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

    # fig 2: underwater (folded-arm) config
    fig2 = plt.figure(figsize=(9, 8))
    ax2  = fig2.add_subplot(111, projection='3d')
    ax2.set_title("UAUV Underwater Geometry  (arms folded)", fontsize=13, fontweight='bold')

    info_uw = (f"Total mass : {total_mass:.2f} kg  (unchanged)\n"
               f"L_fold     : {UW_L_fold:.3f} m  (frac={UW_ARM_FOLD_FRAC})\n"
               f"Ixx_uw : {Ixx_uw:.4f} kg·m²\n"
               f"Iyy_uw : {Iyy_uw:.4f} kg·m²\n"
               f"Izz_uw : {Izz_uw:.4f} kg·m²")
    fig2.text(0.02, 0.02, info_uw, fontsize=9, va='bottom', family='monospace',
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

        # seg 1: box centre to fold joint
        ax2.plot([start[0], fold[0]], [start[1], fold[1]], [start[2], fold[2]],
                 'k-', lw=3)
        # seg 2: fold joint to motor (folded-back)
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

        # Label: offset radially beyond disk edge so it never overlaps the prop
        _mag = np.sqrt(motor[0]**2 + motor[1]**2)
        _ux, _uy = (motor[0] / _mag, motor[1] / _mag) if _mag > 1e-6 else (1.0, 0.0)
        _loff = prop_r_uw * 2.2   # radial clearance past disk edge
        ax2.text(motor[0] + _ux * _loff, motor[1] + _uy * _loff, motor[2] + prop_r_uw * 0.8,
                 motor_labels[i], fontsize=9, ha='center', fontweight='bold')

    ax2.scatter(0, 0, 0, s=200, color='gold', marker='*', zorder=10, label='CoM')

    # horizontal thrusters at box corners
    _alpha_h = np.radians(45.0)   # thrust angle from wall toward interior
    _px = L_box / 2.0 - CoM[0]   # corner x in CoM frame (CoM[0]=0 by symmetry)
    _py = W_box / 2.0 - CoM[1]   # corner y in CoM frame (CoM[1]=0 by symmetry)
    _cz = -CoM[2]                  # corner z in CoM frame (box centre at -CoM[2])
    _arr = L_box * 0.18            # arrow length
    _horiz_corners = [
        ( _px, -_py, _cz, +np.cos(_alpha_h), +np.sin(_alpha_h), 'FR'),
        ( _px,  _py, _cz, +np.cos(_alpha_h), -np.sin(_alpha_h), 'FL'),
        (-_px,  _py, _cz, -np.cos(_alpha_h), -np.sin(_alpha_h), 'RL'),
        (-_px, -_py, _cz, -np.cos(_alpha_h), +np.sin(_alpha_h), 'RR'),
    ]
    for hx, hy, hz, dx, dy, lbl in _horiz_corners:
        # Marker at corner
        ax2.scatter(hx, hy, hz, s=60, color='darkorange', edgecolors='k',
                    linewidth=0.8, zorder=7)
        # Arrow showing thrust direction
        ax2.quiver(hx, hy, hz, dx*_arr, dy*_arr, 0,
                   color='darkorange', arrow_length_ratio=0.35, linewidth=1.5)
        ax2.text(hx + dx*_arr*1.3, hy + dy*_arr*1.3, hz,
                 lbl, fontsize=7, ha='center', color='darkorange', fontweight='bold')

    axis_len2 = UW_L_fold * 0.6
    ax2.quiver(0, 0, 0, axis_len2, 0, 0, color='r', arrow_length_ratio=0.2)
    ax2.quiver(0, 0, 0, 0, axis_len2, 0, color='g', arrow_length_ratio=0.2)
    ax2.quiver(0, 0, 0, 0, 0, axis_len2, color='b', arrow_length_ratio=0.2)
    ax2.text(axis_len2*1.15, 0, 0, 'x', color='r', fontsize=10)
    ax2.text(0, axis_len2*1.15, 0, 'y', color='g', fontsize=10)
    ax2.text(0, 0, axis_len2*1.15, 'z', color='b', fontsize=10)

    span2 = max(L_box, W_box) * 0.7 + _arr * 1.5
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
        plt.Line2D([0],[0], color='darkorange', lw=2,      label='Horiz thruster'),
    ], labels=['CoM', 'CCW rotor', 'CW rotor', 'Arm seg 1', 'Arm seg 2 (folded)', 'Horiz thruster'])

    fig2.tight_layout()
    plt.show()
