import matplotlib.pyplot as plt
import numpy as np

# ── USER INPUTS ──────────────────────────────────────────────────────────────
rho = 1025.0    # [kg/m³]  water density
D   = 0.0762     # [m]      thruster propeller diameter — set your value here
# ─────────────────────────────────────────────────────────────────────────────

# Forward direction (positive force, PWM 1532–1900)
rpm_fwd = np.array([
     308,  375,  441,  508,  573,  630,  688,  749,  804,  859,
     906,  962, 1007, 1057, 1104, 1148, 1195, 1237, 1279, 1319,
    1359, 1409, 1444, 1480, 1516, 1549, 1583, 1629, 1668, 1699,
    1740, 1766, 1802, 1835, 1865, 1901, 1930, 1961, 1994, 2031,
    2071, 2106, 2129, 2160, 2190, 2219, 2244, 2266, 2305, 2342,
    2392, 2416, 2446, 2475, 2498, 2538, 2584, 2609, 2639, 2663,
    2688, 2711, 2735, 2766, 2809, 2822, 2854, 2888, 2903, 2934,
    2973, 3002, 3042, 3065, 3108, 3125, 3153, 3180, 3204, 3242,
    3260, 3299, 3308, 3351, 3379, 3406, 3428, 3438, 3455, 3494,
    3516, 3527, 3533,
    ])

thrust_fwd_kgf = np.array([
    0.04, 0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.22, 0.25, 0.29,
    0.32, 0.36, 0.40, 0.44, 0.47, 0.51, 0.56, 0.60, 0.64, 0.68,
    0.72, 0.78, 0.82, 0.87, 0.91, 0.95, 0.99, 1.03, 1.10, 1.14,
    1.18, 1.24, 1.28, 1.33, 1.39, 1.44, 1.48, 1.54, 1.59, 1.65,
    1.69, 1.76, 1.82, 1.88, 1.93, 1.99, 2.05, 2.12, 2.18, 2.22,
    2.28, 2.38, 2.43, 2.52, 2.58, 2.65, 2.73, 2.76, 2.84, 2.89,
    2.98, 3.05, 3.11, 3.17, 3.22, 3.30, 3.37, 3.42, 3.51, 3.61,
    3.68, 3.74, 3.82, 3.89, 3.96, 4.06, 4.15, 4.25, 4.30, 4.38,
    4.51, 4.53, 4.65, 4.71, 4.79, 4.84, 4.93, 5.01, 5.08, 5.14,
    5.18, 5.22, 5.25,
])

# Reverse direction (negative force by sign, PWM 1100–1468, magnitudes used)
rpm_rev = np.array([
    3465, 3468, 3449, 3421, 3410, 3395, 3374, 3356, 3323, 3307,
    3282, 3248, 3223, 3207, 3177, 3153, 3111, 3086, 3061, 3050,
    3003, 2988, 2952, 2918, 2896, 2865, 2845, 2821, 2792, 2759,
    2736, 2712, 2681, 2661, 2629, 2599, 2576, 2548, 2517, 2482,
    2440, 2419, 2369, 2345, 2306, 2278, 2248, 2222, 2199, 2157,
    2140, 2106, 2081, 2040, 2001, 1968, 1936, 1903, 1870, 1840,
    1807, 1773, 1738, 1705, 1667, 1630, 1589, 1554, 1519, 1482,
    1448, 1410, 1362, 1325, 1285, 1246, 1199, 1153, 1113, 1064,
    1016,  970,  916,  869,  810,  758,  697,  641,  578,  513,
     450,  388,  317,
])

thrust_rev_kgf = np.array([
    4.07, 4.05, 4.02, 3.96, 3.90, 3.87, 3.82, 3.80, 3.75, 3.71,
    3.66, 3.59, 3.52, 3.45, 3.40, 3.31, 3.25, 3.20, 3.11, 3.04,
    2.99, 2.94, 2.86, 2.82, 2.77, 2.71, 2.66, 2.58, 2.55, 2.51,
    2.45, 2.38, 2.35, 2.28, 2.24, 2.20, 2.12, 2.08, 2.02, 1.98,
    1.94, 1.86, 1.81, 1.76, 1.70, 1.67, 1.61, 1.56, 1.51, 1.49,
    1.44, 1.40, 1.35, 1.30, 1.26, 1.20, 1.16, 1.12, 1.10, 1.05,
    1.02, 0.98, 0.94, 0.90, 0.87, 0.82, 0.78, 0.74, 0.72, 0.68,
    0.65, 0.62, 0.58, 0.54, 0.51, 0.48, 0.44, 0.42, 0.39, 0.35,
    0.32, 0.29, 0.26, 0.24, 0.21, 0.18, 0.15, 0.13, 0.10, 0.09,
    0.07, 0.05, 0.04,
])

# ── Derived quantities ────────────────────────────────────────────────────────
omega_fwd = rpm_fwd * (2 * np.pi / 60)       # [rad/s]
n_fwd     = omega_fwd / (2 * np.pi)           # [rev/s]
T_fwd     = (thrust_fwd_kgf / 1000) * 9.81 * 1000   # [N]  kgf → N  (1 kgf = 9.81 N)
# correct: T [N] = kgf * 9.81
T_fwd     = thrust_fwd_kgf * 9.81             # [N]

omega_rev = rpm_rev * (2 * np.pi / 60)
n_rev     = omega_rev / (2 * np.pi)
T_rev     = thrust_rev_kgf * 9.81             # [N]  (magnitudes)

# ── Thrust — forward ──────────────────────────────────────────────────────────
X_T_fwd = rho * n_fwd**2 * D**4
C_T     = (X_T_fwd @ T_fwd) / (X_T_fwd @ X_T_fwd)   # least-squares through origin
kT      = C_T * rho * D**4 / (2 * np.pi)**2           # kT·ω² = T

# ── Thrust — reverse ─────────────────────────────────────────────────────────
X_T_rev  = rho * n_rev**2 * D**4
C_T_rev  = (X_T_rev @ T_rev) / (X_T_rev @ X_T_rev)
kT_rev   = C_T_rev * rho * D**4 / (2 * np.pi)**2

T_fit_fwd = C_T     * X_T_fwd
T_fit_rev = C_T_rev * X_T_rev

if __name__ == "__main__":
    print(f"Propeller diameter : {D*100:.1f} cm")
    print(f"Fluid density      : {rho} kg/m³")
    print(f"Fitted C_T  (fwd)  : {C_T:.6f}")
    print(f"Fitted kT   (fwd)  : {kT:.4e}  [N/(rad/s)²]")
    print(f"Fitted C_T  (rev)  : {C_T_rev:.6f}")
    print(f"Fitted kT   (rev)  : {kT_rev:.4e}  [N/(rad/s)²]")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    n2_line = np.linspace(0, max(n_fwd.max(), n_rev.max())**2, 200)

    R2_fwd = 1 - np.sum((T_fwd - T_fit_fwd)**2) / np.sum((T_fwd - T_fwd.mean())**2)
    R2_rev = 1 - np.sum((T_rev - T_fit_rev)**2) / np.sum((T_rev - T_rev.mean())**2)

    ax = axes[0]
    ax.plot(n_fwd**2, T_fwd, 'o', color='steelblue', label='Measured (fwd)', markersize=4)
    ax.plot(n2_line, C_T * rho * n2_line * D**4, '-', color='tomato', linewidth=2,
            label=f'Fit  $C_T$={C_T:.4f}  $R^2$={R2_fwd:.5f}')
    ax.set_xlabel('n²  [rev²/s²]'); ax.set_ylabel('T  [N]')
    ax.set_title('Forward Thrust  (T = $C_T$ ρ n² D⁴)'); ax.legend(); ax.grid(True)

    ax = axes[1]
    ax.plot(n_rev**2, T_rev, 's', color='seagreen', label='Measured (rev)', markersize=4)
    ax.plot(n2_line, C_T_rev * rho * n2_line * D**4, '-', color='darkorange', linewidth=2,
            label=f'Fit  $C_T$={C_T_rev:.4f}  $R^2$={R2_rev:.5f}')
    ax.set_xlabel('n²  [rev²/s²]'); ax.set_ylabel('T  [N]  (magnitude)')
    ax.set_title('Reverse Thrust  (T = $C_T$ ρ n² D⁴)'); ax.legend(); ax.grid(True)

    plt.tight_layout()

    # ── RPM vs Thrust (single axis, negative RPM = reverse) ──────────────────
    # Hover operating point — edit to match your vehicle
    L_box, W_box, H_box = 0.8, 0.4, 0.4   # [m]  box dimensions from geometry.py
    m_total           = 25.0               # [kg] total mass
    BUOYANCY_COMP_PCT = 95.0               # [%]  ballast flooding
    N_THRUSTERS       = 4

    F_net     = rho * L_box * W_box * H_box * 9.81 - m_total * 9.81
    F_resid   = (1.0 - BUOYANCY_COMP_PCT / 100.0) * F_net
    T_hover   = F_resid / N_THRUSTERS
    rpm_hover = np.sqrt(T_hover / kT_rev) * 60 / (2 * np.pi)

    rpm_line = np.linspace(0, max(rpm_fwd.max(), rpm_rev.max()), 400)
    n_line   = rpm_line / 60.0
    T_fit_line_fwd = C_T     * rho * n_line**2 * D**4
    T_fit_line_rev = C_T_rev * rho * n_line**2 * D**4

    fig2, ax2 = plt.subplots(figsize=(11, 6))

    ax2.scatter( rpm_fwd,  T_fwd,  color='steelblue',  s=18, zorder=5, label='Measured — forward')
    ax2.scatter(-rpm_rev, -T_rev,  color='seagreen',   s=18, zorder=5, label='Measured — reverse')
    ax2.plot( rpm_line,  T_fit_line_fwd, '-', color='tomato',     lw=2,
              label=f'Fit fwd   kT={kT:.3e} N·s²/rad²')
    ax2.plot(-rpm_line, -T_fit_line_rev, '-', color='darkorange', lw=2,
              label=f'Fit rev   kT={kT_rev:.3e} N·s²/rad²')

    # Hover point: reverse thrust cancels residual upward buoyancy
    ax2.scatter(-rpm_hover, -T_hover, color='red', s=180, zorder=10, marker='*',
                label=f'Hover  −{rpm_hover:.0f} RPM  →  {T_hover:.1f} N/thruster × {N_THRUSTERS} = {F_resid:.0f} N')
    ax2.axhline(-T_hover,   color='red', lw=0.8, ls=':')
    ax2.axvline(-rpm_hover, color='red', lw=0.8, ls=':')
    ax2.annotate(f'{rpm_hover:.0f} RPM\n{T_hover:.1f} N/thr',
                 xy=(-rpm_hover, -T_hover), xytext=(-rpm_hover - 250, -T_hover - 8),
                 fontsize=9, color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    ax2.axhline(0, color='k', lw=0.8, ls='--')
    ax2.axvline(0, color='k', lw=0.8, ls='--')
    ax2.set_xlabel('RPM  (+ forward / − reverse)', fontsize=12)
    ax2.set_ylabel('Thrust  [N]  (+ forward / − reverse)', fontsize=12)
    ax2.set_title(f'RPM vs Thrust  —  D={D*100:.1f} cm  ρ={rho} kg/m³\n'
                  f'Net buoyancy: {F_net:.0f} N  |  {BUOYANCY_COMP_PCT:.0f}% ballast  '
                  f'→  {F_resid:.0f} N residual for thrusters', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True)
    fig2.tight_layout()

    plt.show()
