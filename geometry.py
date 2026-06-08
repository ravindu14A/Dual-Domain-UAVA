# UAUV geometry and mass model
# pill-shaped (capsule) hull + 4 arms (X-config) + 4 motor point masses
# body frame: x forward, y left, z up, origin at CoM
# pill axis along x: two hemispheres of radius R_pill capping a cylinder of length L_cyl

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- parameters (edit these) ---

# pill hull
R_pill = 0.1    # [m]  cross-section radius  (diameter = 0.4 m)
L_cyl  = 0.7    # [m]  length of cylindrical section — sized for +50 N buoyancy at 25.6 kg
m_hull = 21    # [kg] hull + all internal mass (electronics, payload, flooded water, etc.)

# 4 arms in X-config, 45 deg offsets from hull top
L_arm   = 0.7   # [m]
m_arm   = 0.15   # [kg] each
m_motor = 0.50   # [kg] per motor (point mass at arm tip)

# underwater fold fraction: fold joint at frac*L_arm from base
# after 180 deg fold: L_fold = (2f-1)*L_arm  (must be > 0.5)
UW_ARM_FOLD_FRAC = 0.75

# --- derived geometry ---

# backward-compat shims used by quadcopterSC, quadcopterUW, Test_time, etc.
L_box = L_cyl + 2.0 * R_pill   # [m] total pill length (= bounding-box x)
W_box = 2.0 * R_pill            # [m] pill diameter (bounding-box y)
H_box = 2.0 * R_pill            # [m] pill diameter (bounding-box z)

# pill volume: cylinder + two hemispheres
V_pill = np.pi * R_pill**2 * L_cyl + (4.0 / 3.0) * np.pi * R_pill**3

# arm angles: FL=45, RL=135, RR=225, FR=315
arm_angles_deg = [45.0, 135.0, 225.0, 315.0]
arm_angles_rad = np.radians(arm_angles_deg)

z_attach = R_pill   # arms attach at the top of the circular cross-section

arm_com = np.array([
    [(L_arm / 2) * np.cos(a), (L_arm / 2) * np.sin(a), z_attach]
    for a in arm_angles_rad
])
motor_pos = np.array([
    [L_arm * np.cos(a), L_arm * np.sin(a), z_attach]
    for a in arm_angles_rad
])

# --- total mass and CoM ---

total_mass = m_hull + 4 * m_arm + 4 * m_motor

com_x = (m_hull * 0.0 + m_arm * arm_com[:, 0].sum() + m_motor * motor_pos[:, 0].sum()) / total_mass
com_y = (m_hull * 0.0 + m_arm * arm_com[:, 1].sum() + m_motor * motor_pos[:, 1].sum()) / total_mass
com_z = (m_hull * 0.0 + m_arm * arm_com[:, 2].sum() + m_motor * motor_pos[:, 2].sum()) / total_mass
CoM = np.array([com_x, com_y, com_z])

arm_com_c   = arm_com   - CoM
motor_pos_c = motor_pos - CoM

# --- pill hull moments of inertia ---
# Capsule axis along x.  Cylinder has length L_cyl, both hemispheres radius R_pill.
# Hull mass split proportionally to volume (uniform density assumption).
_V_cyl  = np.pi * R_pill**2 * L_cyl
_V_hemi = (2.0 / 3.0) * np.pi * R_pill**3   # one hemisphere
m_cyl   = m_hull * _V_cyl  / V_pill
m_hemi  = m_hull * _V_hemi / V_pill          # mass of ONE hemisphere

# Cylinder about its own centre:
#   I_xx = m/2  * R^2          (along axis)
#   I_yy = m/12 * (3R^2+L^2)  (transverse)
_I_cyl_xx = m_cyl / 2.0 * R_pill**2
_I_cyl_yy = m_cyl / 12.0 * (3.0 * R_pill**2 + L_cyl**2)

# Each hemisphere about ITS OWN CoM (CoM is at 3R/8 from flat face):
#   I_sym  = 2/5  * m * R^2          (about symmetry / x axis)
#   I_tran = 83/320 * m * R^2        (about transverse axis through CoM)
# Then parallel-axis to capsule centre: d = L_cyl/2 + 3R/8
_d_hemi = L_cyl / 2.0 + 3.0 * R_pill / 8.0
_I_hemi_xx = (2.0 / 5.0) * m_hemi * R_pill**2
_I_hemi_yy = (83.0 / 320.0) * m_hemi * R_pill**2 + m_hemi * _d_hemi**2

# Two hemispheres combined:
_I_h2_xx = 2.0 * _I_hemi_xx
_I_h2_yy = 2.0 * _I_hemi_yy

Ixx_hull = _I_cyl_xx + _I_h2_xx
Iyy_hull = _I_cyl_yy + _I_h2_yy
Izz_hull = Iyy_hull   # symmetric capsule: I_yy == I_zz

# --- arm + motor inertia (aerial configuration, arms extended) ---
Ixx_arms = Iyy_arms = Izz_arms = 0.0
for i, a in enumerate(arm_angles_rad):
    cx, cy, cz = arm_com_c[i]
    ux, uy = np.cos(a), np.sin(a)
    I_rod_x = m_arm * L_arm**2 / 12.0 * uy**2
    I_rod_y = m_arm * L_arm**2 / 12.0 * ux**2
    I_rod_z = m_arm * L_arm**2 / 12.0
    Ixx_arms += I_rod_x + m_arm * (cy**2 + cz**2)
    Iyy_arms += I_rod_y + m_arm * (cx**2 + cz**2)
    Izz_arms += I_rod_z + m_arm * (cx**2 + cy**2)

Ixx_motors = sum(m_motor * (r[1]**2 + r[2]**2) for r in motor_pos_c)
Iyy_motors = sum(m_motor * (r[0]**2 + r[2]**2) for r in motor_pos_c)
Izz_motors = sum(m_motor * (r[0]**2 + r[1]**2) for r in motor_pos_c)

Ixx = Ixx_hull + Ixx_arms + Ixx_motors
Iyy = Iyy_hull + Iyy_arms + Iyy_motors
Izz = Izz_hull + Izz_arms + Izz_motors

# --- underwater inertia (folded-arm config) ---
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

Ixx_arms_uw = Iyy_arms_uw = Izz_arms_uw = 0.0
for i, a in enumerate(arm_angles_rad):
    ux, uy = np.cos(a), np.sin(a)
    for m_seg, L_seg, (cx, cy, cz) in [
        (_uw_m1, _uw_L1, uw_seg1_com_c[i]),
        (_uw_m2, _uw_L2, uw_seg2_com_c[i]),
    ]:
        I_x = m_seg * L_seg**2 / 12.0 * uy**2
        I_y = m_seg * L_seg**2 / 12.0 * ux**2
        I_z = m_seg * L_seg**2 / 12.0
        Ixx_arms_uw += I_x + m_seg * (cy**2 + cz**2)
        Iyy_arms_uw += I_y + m_seg * (cx**2 + cz**2)
        Izz_arms_uw += I_z + m_seg * (cx**2 + cy**2)

Ixx_motors_uw = sum(m_motor * (r[1]**2 + r[2]**2) for r in uw_motor_pos_c)
Iyy_motors_uw = sum(m_motor * (r[0]**2 + r[2]**2) for r in uw_motor_pos_c)
Izz_motors_uw = sum(m_motor * (r[0]**2 + r[1]**2) for r in uw_motor_pos_c)

Ixx_uw = Ixx_hull + Ixx_arms_uw + Ixx_motors_uw
Iyy_uw = Iyy_hull + Iyy_arms_uw + Iyy_motors_uw
Izz_uw = Izz_hull + Izz_arms_uw + Izz_motors_uw


def _draw_capsule(ax, cx, cy, cz, R, L, color='steelblue', alpha=0.35, n=40):
    """Draw a capsule (cylinder + 2 hemispherical caps) along the x-axis."""
    u = np.linspace(0, 2 * np.pi, n)
    # Cylinder
    z_cyl = np.array([-L / 2, L / 2])
    U, Z = np.meshgrid(u, z_cyl)
    Xc = cx + Z
    Yc = cy + R * np.cos(U)
    Zc = cz + R * np.sin(U)
    ax.plot_surface(Xc, Yc, Zc, color=color, alpha=alpha, linewidth=0, antialiased=True)
    # Hemisphere caps  (v in [0, pi/2])
    v = np.linspace(0, np.pi / 2, n // 2)
    U2, V2 = np.meshgrid(u, v)
    # Front cap (+x)
    ax.plot_surface(cx + L / 2 + R * np.sin(V2),
                    cy + R * np.cos(U2) * np.cos(V2),
                    cz + R * np.sin(U2) * np.cos(V2),
                    color=color, alpha=alpha, linewidth=0, antialiased=True)
    # Rear cap (−x)
    ax.plot_surface(cx - L / 2 - R * np.sin(V2),
                    cy + R * np.cos(U2) * np.cos(V2),
                    cz + R * np.sin(U2) * np.cos(V2),
                    color=color, alpha=alpha, linewidth=0, antialiased=True)


if __name__ == "__main__":
    print("=" * 52)
    print("UAUV MASS PROPERTIES  (pill hull)")
    print("=" * 52)
    print(f"  Hull (pill)     : {m_hull:.3f} kg   R={R_pill:.3f} m  L_cyl={L_cyl:.3f} m")
    print(f"  Total length    : {L_box:.3f} m  (L_cyl + 2R)")
    print(f"  Hull volume     : {V_pill*1e3:.2f} L  ({V_pill:.5f} m³)")
    print(f"  Arms (x4)       : {4*m_arm:.3f} kg   ({m_arm:.3f} kg each, L={L_arm:.3f} m)")
    print(f"  Motors (x4)     : {4*m_motor:.3f} kg   ({m_motor:.3f} kg each)")
    print(f"  TOTAL MASS      : {total_mass:.3f} kg")
    print()
    print(f"  Centre of Mass  : [{CoM[0]:+.4f},  {CoM[1]:+.4f},  {CoM[2]:+.4f}] m")
    print()
    print(f"  Ixx (roll)      : {Ixx:.5f} kg·m²")
    print(f"  Iyy (pitch)     : {Iyy:.5f} kg·m²")
    print(f"  Izz (yaw)       : {Izz:.5f} kg·m²")
    print("=" * 52)
    print()
    print(f"  UW Ixx          : {Ixx_uw:.5f} kg·m²")
    print(f"  UW Iyy          : {Iyy_uw:.5f} kg·m²")
    print(f"  UW Izz          : {Izz_uw:.5f} kg·m²")
    print(f"  L_fold          : {UW_L_fold:.3f} m  (frac={UW_ARM_FOLD_FRAC})")
    print("=" * 52)

    _ang = np.linspace(0, 2 * np.pi, 40)
    motor_labels = ['FL', 'RL', 'RR', 'FR']
    spin_color   = ['tomato', 'royalblue', 'tomato', 'royalblue']
    prop_r       = L_arm * 0.12

    # ── fig 1: aerial config ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title("UAUV Geometry — Aerial (arms extended)", fontsize=13, fontweight='bold')

    fig.text(0.02, 0.97,
             f"Total mass : {total_mass:.2f} kg\n"
             f"Ixx : {Ixx:.4f} kg·m²\n"
             f"Iyy : {Iyy:.4f} kg·m²\n"
             f"Izz : {Izz:.4f} kg·m²\n"
             f"V_pill : {V_pill*1e3:.2f} L",
             fontsize=9, va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    _draw_capsule(ax, -CoM[0], -CoM[1], -CoM[2], R_pill, L_cyl,
                  color='steelblue', alpha=0.40)

    for i in range(4):
        start = np.array([0.0, 0.0, z_attach]) - CoM
        end   = motor_pos_c[i]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'k-', lw=3)
        cx, cy, cz = motor_pos_c[i]
        xd = cx + prop_r * np.cos(_ang)
        yd = cy + prop_r * np.sin(_ang)
        disk = Poly3DCollection([list(zip(xd, yd, np.full(40, cz)))],
                                alpha=0.75, facecolor=spin_color[i], edgecolor='k', linewidth=0.8)
        ax.add_collection3d(disk)
        ax.text(motor_pos_c[i, 0]*1.12, motor_pos_c[i, 1]*1.12, motor_pos_c[i, 2],
                motor_labels[i], fontsize=9, ha='center')

    ax.scatter(0, 0, 0, s=200, color='gold', marker='*', zorder=10, label='CoM')
    axis_len = L_arm * 0.4
    for vec, col, lbl in [(np.array([1,0,0]),'r','x'),
                           (np.array([0,1,0]),'g','y'),
                           (np.array([0,0,1]),'b','z')]:
        ax.quiver(0,0,0,*(axis_len*vec), color=col, arrow_length_ratio=0.2)
        ax.text(*(axis_len*1.15*vec), lbl, color=col, fontsize=10)

    span = L_arm * 1.4
    ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_zlim(-span*0.6, span)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    ax.set_box_aspect([1, 1, 1])

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        ax.get_legend_handles_labels()[0][0],
        mpatches.Patch(facecolor='tomato',    edgecolor='k', label='CCW rotor'),
        mpatches.Patch(facecolor='royalblue', edgecolor='k', label='CW rotor'),
    ], labels=['CoM', 'CCW rotor', 'CW rotor'])
    plt.tight_layout()

    # ── fig 2: underwater config (arms folded) ───────────────────────────────
    fig2 = plt.figure(figsize=(9, 8))
    ax2  = fig2.add_subplot(111, projection='3d')
    ax2.set_title("UAUV Geometry — Underwater (arms folded)", fontsize=13, fontweight='bold')

    fig2.text(0.02, 0.02,
              f"Total mass : {total_mass:.2f} kg  (unchanged)\n"
              f"L_fold     : {UW_L_fold:.3f} m  (frac={UW_ARM_FOLD_FRAC})\n"
              f"Ixx_uw : {Ixx_uw:.4f} kg·m²\n"
              f"Iyy_uw : {Iyy_uw:.4f} kg·m²\n"
              f"Izz_uw : {Izz_uw:.4f} kg·m²",
              fontsize=9, va='bottom', family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    _draw_capsule(ax2, -CoM[0], -CoM[1], -CoM[2], R_pill, L_cyl,
                  color='steelblue', alpha=0.40)

    prop_r_uw = UW_L_fold * 0.18
    for i in range(4):
        a     = arm_angles_rad[i]
        fold  = np.array([UW_ARM_FOLD_FRAC * L_arm * np.cos(a),
                          UW_ARM_FOLD_FRAC * L_arm * np.sin(a), z_attach]) - CoM
        start = np.array([0.0, 0.0, z_attach]) - CoM
        motor = uw_motor_pos_c[i]
        ax2.plot([start[0], fold[0]], [start[1], fold[1]], [start[2], fold[2]], 'k-', lw=3)
        ax2.plot([fold[0], motor[0]], [fold[1], motor[1]], [fold[2], motor[2]],
                 color='gray', lw=2, ls='--')
        ax2.scatter(*fold, s=60, color='black', zorder=6)
        cx, cy, cz = motor
        xd = cx + prop_r_uw * np.cos(_ang)
        yd = cy + prop_r_uw * np.sin(_ang)
        disk2 = Poly3DCollection([list(zip(xd, yd, np.full(40, cz)))],
                                 alpha=0.75, facecolor=spin_color[i], edgecolor='k', linewidth=0.8)
        ax2.add_collection3d(disk2)
        _mag = np.sqrt(motor[0]**2 + motor[1]**2)
        _ux, _uy = (motor[0]/_mag, motor[1]/_mag) if _mag > 1e-6 else (1.0, 0.0)
        ax2.text(motor[0] + _ux*prop_r_uw*2.2, motor[1] + _uy*prop_r_uw*2.2, motor[2]+prop_r_uw*0.8,
                 motor_labels[i], fontsize=9, ha='center', fontweight='bold')

    ax2.scatter(0, 0, 0, s=200, color='gold', marker='*', zorder=10, label='CoM')

    # horizontal thrusters at hull corners (approximate: at ±L_cyl/2 ± R in y)
    _alpha_h = np.radians(45.0)
    _px = L_cyl / 2.0 - CoM[0]
    _py = R_pill - CoM[1]
    _cz = -CoM[2]
    _arr = L_box * 0.18
    for hx, hy, hz, dx, dy, lbl in [
        ( _px, -_py, _cz, +np.cos(_alpha_h), +np.sin(_alpha_h), 'FR'),
        ( _px,  _py, _cz, +np.cos(_alpha_h), -np.sin(_alpha_h), 'FL'),
        (-_px,  _py, _cz, -np.cos(_alpha_h), -np.sin(_alpha_h), 'RL'),
        (-_px, -_py, _cz, -np.cos(_alpha_h), +np.sin(_alpha_h), 'RR'),
    ]:
        ax2.scatter(hx, hy, hz, s=60, color='darkorange', edgecolors='k', linewidth=0.8, zorder=7)
        ax2.quiver(hx, hy, hz, dx*_arr, dy*_arr, 0,
                   color='darkorange', arrow_length_ratio=0.35, linewidth=1.5)
        ax2.text(hx+dx*_arr*1.3, hy+dy*_arr*1.3, hz, lbl,
                 fontsize=7, ha='center', color='darkorange', fontweight='bold')

    axis_len2 = UW_L_fold * 0.6
    for vec, col, lbl in [(np.array([1,0,0]),'r','x'),
                           (np.array([0,1,0]),'g','y'),
                           (np.array([0,0,1]),'b','z')]:
        ax2.quiver(0,0,0,*(axis_len2*vec), color=col, arrow_length_ratio=0.2)
        ax2.text(*(axis_len2*1.15*vec), lbl, color=col, fontsize=10)

    span2 = max(L_box, W_box) * 0.7 + _arr * 1.5
    ax2.set_xlim(-span2, span2); ax2.set_ylim(-span2, span2); ax2.set_zlim(-span2, span2)
    ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]'); ax2.set_zlabel('z [m]')
    ax2.set_box_aspect([1, 1, 1])

    ax2.legend(handles=[
        ax2.get_legend_handles_labels()[0][0],
        mpatches.Patch(facecolor='tomato',    edgecolor='k', label='CCW rotor'),
        mpatches.Patch(facecolor='royalblue', edgecolor='k', label='CW rotor'),
        plt.Line2D([0],[0], color='k',    lw=3,          label='Arm seg 1'),
        plt.Line2D([0],[0], color='gray', lw=2, ls='--', label='Arm seg 2 (folded)'),
        plt.Line2D([0],[0], color='darkorange', lw=2,    label='Horiz thruster'),
    ], labels=['CoM', 'CCW rotor', 'CW rotor', 'Arm seg 1', 'Arm seg 2 (folded)', 'Horiz thruster'])

    fig2.tight_layout()
    plt.show()
