"""
Pole-placement gain calculator for cascaded PID controllers.

Produces Kp and Kd for the inner (attitude) and outer (position) loops
of quadcopterSC and quadcopterUW.  Copy-paste the printed output into
the _g = dict(...) block in each file.

Inertia is pulled directly from geometry.py — set USE_CATIA_INERTIA
there before running this if you want CATIA values.

Inner loop  plant: tau = I * angle_ddot  ->  P(s) = 1 / (I * s^2)
  Kp = I * omega_n^2
  Kd = I * 2 * zeta * omega_n

Outer loop  plant: x_ddot = a_cmd  ->  P(s) = 1 / s^2
  (inner loop treated as unity — valid when omega_n_att >> omega_n_cyl)
  Kp = omega_n^2
  Kd = 2 * zeta * omega_n

Axes order throughout: [roll/phi,  pitch/theta,  yaw/psi]   (att)
                       [r,         tangential,   z      ]   (cyl)

Ki and integral/output limits are NOT computed here — keep the existing
values from the file or tune them by hand after.
"""

import numpy as np
from geometry import (Ixx, Iyy, Izz, Ixx_uw, Iyy_uw, Izz_uw,
                      USE_CATIA_INERTIA, LIN_POS, LIN_EULER, LIN_VEL)

# ─────────────────────────────────────────────────────────────────────────────
#  SC (AERIAL) TUNING PARAMETERS  — edit these
# ─────────────────────────────────────────────────────────────────────────────

# Inner (attitude) loop — [roll, pitch, yaw]
SC_ATT_OMEGA_N = np.array([4.0,  8.0,  4.0])   # [rad/s] natural frequency per axis
SC_ATT_ZETA    = np.array([0.9,  0.9,  0.9])   # damping ratio per axis

# Outer (position) loop — [r, tangential, z]
SC_CYL_OMEGA_N = np.array([0.5,  0.5,  1.2])   # [rad/s] — keep << inner omega_n
SC_CYL_ZETA    = np.array([0.9,  0.9,  0.9])

# Ki values to carry over unchanged (pole placement doesn't compute these)
SC_ATT_KI    = np.array([0.1,   0.1,    0.05])
SC_ATT_I_LIM = np.array([5.0,   5.0,    3.0 ])
SC_ATT_LIM   = 0.45
SC_CYL_KI    = np.array([0.005, 0.005,  0.05])
SC_CYL_I_LIM = np.array([5.0,   5.0,   10.0 ])

# ─────────────────────────────────────────────────────────────────────────────
#  UW TUNING PARAMETERS  — edit these
# ─────────────────────────────────────────────────────────────────────────────

# Inner (attitude) loop — [roll, pitch, yaw]
UW_ATT_OMEGA_N = np.array([3.0,  4.0,  5.0])
UW_ATT_ZETA    = np.array([0.9,  0.9,  0.9])

# Outer (position) loop — [r, tangential, z]
UW_CYL_OMEGA_N = np.array([0.5,  0.7,  0.7])
UW_CYL_ZETA    = np.array([0.9,  0.9,  0.9])

# Ki values to carry over unchanged
UW_ATT_KI    = np.array([0.03,  0.03,  0.03])
UW_ATT_I_LIM = np.array([2.0,   2.0,   2.0 ])
UW_ATT_LIM   = 0.25
UW_CYL_KI    = np.array([0.002, 0.002, 0.006])
UW_CYL_I_LIM = np.array([3.0,   3.0,   8.0 ])

# ─────────────────────────────────────────────────────────────────────────────
#  COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def place_inner(I_vec, omega_n, zeta):
    """PD gains for attitude loop on double-integrator plant scaled by I."""
    Kp = I_vec * omega_n ** 2
    Kd = I_vec * 2 * zeta * omega_n
    return Kp, Kd

def place_outer(omega_n, zeta):
    """PD gains for position loop on unit double-integrator (a_cmd -> pos)."""
    Kp = omega_n ** 2
    Kd = 2 * zeta * omega_n
    return Kp, Kd

def fmt_arr(arr, width=7):
    return "[" + ",".join(f"{v:{width}.4g}" for v in arr) + "]"

def check_separation(omega_att, omega_cyl, label):
    ratio = omega_att / omega_cyl
    for i, (wa, wc, r) in enumerate(zip(omega_att, omega_cyl, ratio)):
        axis = ["r/roll", "tang/pitch", "z/yaw"][i]
        flag = "  <<< WARNING: too close, cascade assumption may break" if r < 5 else ""
        print(f"    [{axis}]  att={wa:.2f} / cyl={wc:.2f} rad/s  (ratio {r:.1f}x){flag}")

def print_gains(label, att_Kp, att_Kd, att_Ki, att_i_lim, att_lim,
                cyl_Kp, cyl_Kd, cyl_Ki, cyl_i_lim):
    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  _g = dict(")
    print(f"      att_Kp    = np.array({fmt_arr(att_Kp)}),")
    print(f"      att_Ki    = np.array({fmt_arr(att_Ki)}),")
    print(f"      att_Kd    = np.array({fmt_arr(att_Kd)}),")
    print(f"      att_i_lim = np.array({fmt_arr(att_i_lim)}),")
    print(f"      att_lim   = {att_lim},")
    print(f"      cyl_Kp    = np.array({fmt_arr(cyl_Kp)}),")
    print(f"      cyl_Ki    = np.array({fmt_arr(cyl_Ki)}),")
    print(f"      cyl_Kd    = np.array({fmt_arr(cyl_Kd)}),")
    print(f"      cyl_i_lim = np.array({fmt_arr(cyl_i_lim)}),")
    print(f"  )")
    print(sep)

# ─────────────────────────────────────────────────────────────────────────────
#  SC
# ─────────────────────────────────────────────────────────────────────────────

I_sc = np.array([Ixx, Iyy, Izz])
sc_att_Kp, sc_att_Kd = place_inner(I_sc, SC_ATT_OMEGA_N, SC_ATT_ZETA)
sc_cyl_Kp, sc_cyl_Kd = place_outer(SC_CYL_OMEGA_N, SC_CYL_ZETA)

print(f"\nInertia source     : {'CATIA' if USE_CATIA_INERTIA else 'Analytical'}")
print(f"Linearisation point: pos={LIN_POS}  euler={LIN_EULER}  vel={LIN_VEL}")
print(f"  (edit LIN_* in geometry.py to change — applies to root locus too)")
print(f"SC  Ixx={Ixx:.4f}  Iyy={Iyy:.4f}  Izz={Izz:.4f}  [kg·m²]")

print("\nSC bandwidth separation check:")
check_separation(SC_ATT_OMEGA_N, SC_CYL_OMEGA_N, "SC")

print_gains(
    "quadcopterSC  (_g block)",
    sc_att_Kp, sc_att_Kd, SC_ATT_KI, SC_ATT_I_LIM, SC_ATT_LIM,
    sc_cyl_Kp, sc_cyl_Kd, SC_CYL_KI, SC_CYL_I_LIM,
)

# ─────────────────────────────────────────────────────────────────────────────
#  UW
# ─────────────────────────────────────────────────────────────────────────────

I_uw = np.array([Ixx_uw, Iyy_uw, Izz_uw])
uw_att_Kp, uw_att_Kd = place_inner(I_uw, UW_ATT_OMEGA_N, UW_ATT_ZETA)
uw_cyl_Kp, uw_cyl_Kd = place_outer(UW_CYL_OMEGA_N, UW_CYL_ZETA)

print(f"\nUW  Ixx={Ixx_uw:.4f}  Iyy={Iyy_uw:.4f}  Izz={Izz_uw:.4f}  [kg·m²]  (always analytical)")

print("\nUW bandwidth separation check:")
check_separation(UW_ATT_OMEGA_N, UW_CYL_OMEGA_N, "UW")

print_gains(
    "quadcopterUW  (_g block)",
    uw_att_Kp, uw_att_Kd, UW_ATT_KI, UW_ATT_I_LIM, UW_ATT_LIM,
    uw_cyl_Kp, uw_cyl_Kd, UW_CYL_KI, UW_CYL_I_LIM,
)
