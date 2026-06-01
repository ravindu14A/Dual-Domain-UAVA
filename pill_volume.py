import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

RHO_WATER = 1000.0   # kg/m³


def pill_volume(R, L):
    return (4/3) * math.pi * R**3 + math.pi * R**2 * L


def submerged_volume(R, L, z_w):
    """Volume below z_w for horizontal pill (axis along X, centred at origin).

    Cross-section at height z:
      cylinder part  : 2L * sqrt(R²-z²)
      two hemispheres: π  * (R²-z²)

    Integrated analytically from -R to z_w.
    """
    if z_w <= -R:
        return 0.0
    if z_w >= R:
        return pill_volume(R, L)

    def F_cyl(z):
        # antiderivative of 2L*sqrt(R²-z²)
        return L * (z * math.sqrt(R**2 - z**2) + R**2 * math.asin(z / R))

    def F_hemi(z):
        # antiderivative of π*(R²-z²)
        return math.pi * (R**2 * z - z**3 / 3)

    return (F_cyl(z_w) - F_cyl(-R)) + (F_hemi(z_w) - F_hemi(-R))


def waterline_z(R, L, mass):
    """Bisect for z_w where displaced volume × ρ_water = mass."""
    V_needed = mass / RHO_WATER
    if V_needed <= 0:
        return -R
    if V_needed >= pill_volume(R, L):
        return R          # sinks
    z_lo, z_hi = -R, R
    for _ in range(60):
        z_mid = (z_lo + z_hi) / 2
        if submerged_volume(R, L, z_mid) < V_needed:
            z_lo = z_mid
        else:
            z_hi = z_mid
    return (z_lo + z_hi) / 2


def pill_surface(R, L, n=100):
    """Pill with long axis along X."""
    theta = np.linspace(0, 2 * np.pi, n)
    x_cyl = np.linspace(-L / 2, L / 2, n)
    X_cyl, Theta = np.meshgrid(x_cyl, theta)
    Y_cyl = R * np.cos(Theta)
    Z_cyl = R * np.sin(Theta)

    phi = np.linspace(0, np.pi / 2, n // 2)
    Theta_h, Phi = np.meshgrid(theta, phi)

    # right hemisphere (x > 0 end)
    X_rhs =  L / 2 + R * np.cos(Phi)
    Y_rhs = R * np.sin(Phi) * np.cos(Theta_h)
    Z_rhs = R * np.sin(Phi) * np.sin(Theta_h)

    # left hemisphere (x < 0 end)
    X_lhs = -L / 2 - R * np.cos(Phi)
    Y_lhs = R * np.sin(Phi) * np.cos(Theta_h)
    Z_lhs = R * np.sin(Phi) * np.sin(Theta_h)

    return (X_cyl, Y_cyl, Z_cyl), (X_rhs, Y_rhs, Z_rhs), (X_lhs, Y_lhs, Z_lhs)


# ── initial values ────────────────────────────────────────────────────────────
R0, L0 = 0.15, 0.65
mass0  = 10.0    # kg

fig = plt.figure(figsize=(9, 8))
ax  = fig.add_axes([0.05, 0.30, 0.9, 0.66], projection="3d")

ax_R    = fig.add_axes([0.15, 0.18, 0.7, 0.03])
ax_L    = fig.add_axes([0.15, 0.12, 0.7, 0.03])
ax_mass = fig.add_axes([0.15, 0.06, 0.7, 0.03])

slider_R    = Slider(ax_R,    "R",          0.05, 0.30,   valinit=R0,    valstep=0.005)
slider_L    = Slider(ax_L,    "L",          0.30, 0.80,   valinit=L0,    valstep=0.005)
slider_mass = Slider(ax_mass, "mass (kg)",  0.1,  200.0,  valinit=mass0, valstep=0.1)


def draw(R, L, mass):
    ax.cla()

    (X_cyl, Y_cyl, Z_cyl), (X_rhs, Y_rhs, Z_rhs), (X_lhs, Y_lhs, Z_lhs) = pill_surface(R, L)
    ax.plot_surface(X_cyl, Y_cyl, Z_cyl, color="steelblue", alpha=0.6)
    ax.plot_surface(X_rhs, Y_rhs, Z_rhs, color="steelblue", alpha=0.6)
    ax.plot_surface(X_lhs, Y_lhs, Z_lhs, color="steelblue", alpha=0.6)

    lim_x = L / 2 + R + 0.1
    lim_r = R + 0.1
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_r, lim_r)
    ax.set_zlim(-lim_r, lim_r)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    V_total = pill_volume(R, L)
    sinks   = (mass / RHO_WATER) >= V_total

    if sinks:
        ax.set_title(
            f"V = {V_total:.4f} m³  |  mass = {mass:.1f} kg  |  SINKS",
            fontsize=11
        )
    else:
        z_w        = waterline_z(R, L, mass)
        h_from_bot = z_w + R          # height from the very bottom of the pill

        # water plane
        xs = np.linspace(-lim_x, lim_x, 2)
        ys = np.linspace(-lim_r,  lim_r, 2)
        Xw, Yw = np.meshgrid(xs, ys)
        Zw = np.full_like(Xw, z_w)
        ax.plot_surface(Xw, Yw, Zw, color="cyan", alpha=0.25)

        # waterline ring on pill surface
        theta_ring = np.linspace(0, 2 * np.pi, 200)
        y_ring = np.sqrt(max(R**2 - z_w**2, 0)) * np.cos(theta_ring)
        z_ring = np.full_like(theta_ring, z_w)
        for x_pos in np.linspace(-L/2, L/2, 8):
            ax.plot([x_pos]*len(theta_ring), y_ring, z_ring,
                    color="navy", lw=0.6, alpha=0.5)

        ax.set_title(
            f"V = {V_total:.4f} m³  |  mass = {mass:.1f} kg  |  "
            f"waterline z = {z_w:.4f} m  |  height from bottom = {h_from_bot:.4f} m",
            fontsize=10
        )

    fig.canvas.draw_idle()


def update(_):
    draw(slider_R.val, slider_L.val, slider_mass.val)


slider_R.on_changed(update)
slider_L.on_changed(update)
slider_mass.on_changed(update)

draw(R0, L0, mass0)
plt.show()
