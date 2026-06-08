import matplotlib.pyplot as plt
import numpy as np

# ── USER INPUTS ──────────────────────────────────────────────────────────────
rho = 1.225      # [kg/m³]  air density (change to 1025 for water)
D   = 0.6096    # [m]      propeller diameter — set your value here
# ─────────────────────────────────────────────────────────────────────────────

rpm = np.array([
    1960, 2254, 2509, 2740, 2964, 3158, 3528, 3705, 3870, 4017,
    4163, 4308, 4460, 4731, 4866, 4960, 5113, 5221, 5339, 5439
])
thrust_g = np.array([
    1499, 2001, 2498, 2997, 3500, 4001, 4998, 5500, 6003, 6500,
    7005, 7504, 7999, 8999, 9505, 10000, 10503, 10994, 11494, 11852
])

torque_Nm = np.array([
    0.49, 0.59, 0.72, 0.81, 0.92, 1.02, 1.15, 1.25, 1.38, 1.46,
    1.55, 1.72, 1.82, 2.10, 2.28, 2.34, 2.49, 2.64, 2.77, 3.25
])

omega = rpm * (2 * np.pi / 60)          # [rad/s]
n     = omega / (2 * np.pi)             # [rev/s]
T     = (thrust_g / 1000) * 9.81        # [N]
Q     = torque_Nm                        # [N·m]

# ── Thrust ────────────────────────────────────────────────────────────────────
X_T = rho * n**2 * D**4
C_T = (X_T @ T) / (X_T @ X_T)          # least-squares through origin
kT  = C_T * rho * D**4 / (2 * np.pi)**2  # algebra: kT·ω² = C_T·ρ·n²·D⁴

# ── Torque ────────────────────────────────────────────────────────────────────
X_Q = rho * n**2 * D**5
C_Q = (X_Q @ Q) / (X_Q @ X_Q)          # least-squares through origin
kQ  = C_Q * rho * D**5 / (2 * np.pi)**2  # algebra: kQ·ω² = C_Q·ρ·n²·D⁵

T_fit = C_T * X_T
Q_fit = C_Q * X_Q

if __name__ == "__main__":
    print(f"Propeller diameter : {D*100} cm")
    print(f"Fluid density      : {rho} kg/m³")
    print(f"Fitted C_T         : {C_T}")
    print(f"Fitted kT          : {kT}  [N/(rad/s)²]")
    print(f"Fitted C_Q         : {C_Q}")
    print(f"Fitted kQ          : {kQ}  [N·m/(rad/s)²]")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    n2_line = np.linspace(0, n.max()**2, 200)
    R2_T = 1 - np.sum((T - T_fit)**2) / np.sum((T - T.mean())**2)
    R2_Q = 1 - np.sum((Q - Q_fit)**2) / np.sum((Q - Q.mean())**2)

    # Row 0: Thrust
    ax = axes[0, 0]
    ax.plot(omega, T, 'o', color='steelblue', label='Measured')
    ax.plot(omega, T_fit, '-', color='tomato', linewidth=2, label=f'Fit  $C_T$ = {C_T:.4f}')
    ax.set_xlabel('ω  [rad/s]'); ax.set_ylabel('T  [N]')
    ax.set_title('Thrust vs Angular Velocity'); ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(n**2, T, 'o', color='steelblue', label='Measured')
    ax.plot(n2_line, C_T * rho * n2_line * D**4, '-', color='tomato', linewidth=2,
            label=f'Fit  $R^2$ = {R2_T:.5f}')
    ax.set_xlabel('n²  [rev²/s²]'); ax.set_ylabel('T  [N]')
    ax.set_title('Linearity check  (T = $C_T$ ρ n² D⁴)'); ax.legend(); ax.grid(True)

    # Row 1: Torque
    ax = axes[1, 0]
    ax.plot(omega, Q, 's', color='seagreen', label='Measured')
    ax.plot(omega, Q_fit, '-', color='darkorange', linewidth=2, label=f'Fit  $C_Q$ = {C_Q:.6f}')
    ax.set_xlabel('ω  [rad/s]'); ax.set_ylabel('Q  [N·m]')
    ax.set_title('Torque vs Angular Velocity'); ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    ax.plot(n**2, Q, 's', color='seagreen', label='Measured')
    ax.plot(n2_line, C_Q * rho * n2_line * D**5, '-', color='darkorange', linewidth=2,
            label=f'Fit  $R^2$ = {R2_Q:.5f}')
    ax.set_xlabel('n²  [rev²/s²]'); ax.set_ylabel('Q  [N·m]')
    ax.set_title('Linearity check  (Q = $C_Q$ ρ n² D⁵)'); ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.show()