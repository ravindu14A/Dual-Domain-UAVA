# obstacle_avoidance_apf.py
# True Artificial Potential Field (APF) obstacle avoidance.
#
# Returns a repulsive ACCELERATION [m/s²] evaluated at the DRONE'S ACTUAL
# POSITION. Inject directly into a_cmd after the outer position PID but before
# theta_d / phi_d are computed. The position integrator is never disturbed.
#
# Contrast with obstacle_avoidance.py (apf_ref mode):
#   apf_ref  — deflects the reference position; PID must chase the shifted
#               target, so avoidance is gated through the position gains and
#               the integrator accumulates the correction.
#   apf      — adds the gradient directly to a_cmd; immediate effect on thrust
#               and attitude setpoints, integrator is undisturbed.
#
# Tuning guide
# ────────────
# R_BASE       : hard minimum clearance from obstacle surface (m).
# K_APF        : repulsive gain [m³/s²].  Force ∝ K_APF*(1/d−1/d₀)/d².
# R_INFLUENCE  : distance from obstacle surface where field activates (m).
#                At speed v the effective distance = R_INFLUENCE + K_V*v.
# MAX_ACC      : hard cap on total repulsive acceleration [m/s²].
# K_V          : velocity scaling [s] — faster flight triggers avoidance earlier.
# SLOW_FACTOR  : min speed fraction near obstacles (braking channel).

import numpy as np

# ── OBSTACLE DEFINITIONS ─────────────────────────────────────────────────────
#  Keep in sync with obstacle_avoidance.py.
#  Each entry: {'pos': [r_m, theta_deg, z_m], 'radius': float}
OBSTACLES = [
    {'pos': [6, 2, 20], 'radius': 0.3},
]

# ── TUNING PARAMETERS ────────────────────────────────────────────────────────
R_BASE      = 1.5   # [m]     hard clearance — same physical meaning as apf_ref
K_APF       = 2.0   # [m³/s²] repulsive gain (acceleration units, not position)
K_TANG      = 1.5   # [m³/s²] tangential gain — goal-seeking: steers drone around
                    #          obstacle toward the reference regardless of current velocity
K_V         = 0.35  # [s]     velocity scaling
R_INFLUENCE = 5.0   # [m]     base activation distance from obstacle surface
MAX_ACC     = 5.0   # [m/s²]  hard cap — ~0.5 g lateral, leaves margin for control
SLOW_FACTOR = 0.55  # [—]     min speed fraction near obstacle


def _cyl_to_xyz(pos_cyl):
    """[r, theta_deg, z] → [x, y, z]."""
    r, th, z = float(pos_cyl[0]), float(pos_cyl[1]), float(pos_cyl[2])
    return np.array([r * np.cos(np.radians(th)), r * np.sin(np.radians(th)), z])


def compute_apf_acceleration(drone_xyz, vel_vec, ref_xyz=None):
    """
    APF repulsive + goal-seeking tangential acceleration at the drone's position.

    drone_xyz : (3,) drone Cartesian position [m]
    vel_vec   : (3,) drone velocity [m/s]
    ref_xyz   : (3,) current reference position in Cartesian [m] — used to
                compute a goal-seeking tangential that works even when the
                drone is climbing vertically with near-zero horizontal velocity.

    Returns (3,) acceleration [m/s²] to add to a_cmd.
    """
    drone_xyz = np.asarray(drone_xyz, dtype=float)
    vel_vec   = np.asarray(vel_vec,   dtype=float)
    vel_mag   = np.linalg.norm(vel_vec)

    a_rep = np.zeros(3)
    r_act = R_INFLUENCE + K_V * vel_mag

    for obs in OBSTACLES:
        p_obs = _cyl_to_xyz(obs['pos'])
        r_obs = float(obs['radius'])

        diff = drone_xyz - p_obs
        d    = np.linalg.norm(diff)
        if d < 1e-6:
            diff = np.array([1., 0., 0.]); d = 1e-6

        d_surf  = max(d - (r_obs + R_BASE), 1e-3)
        rad_hat = diff / d                        # unit vector: obs → drone

        if d_surf < r_act:
            mag = K_APF * (1.0 / d_surf - 1.0 / r_act) / d_surf**2
            a_rep += mag * rad_hat

            # Goal-seeking tangential — perpendicular component of (drone→ref)
            # relative to the repulsion axis projected into the horizontal plane.
            # Works at any velocity including vertical-only climb.
            # Tangential: rotate repulsion 90° horizontally — always full strength,
            # never collapses to zero even when goal is directly behind obstacle.
            rep_horiz = np.array([rad_hat[0], rad_hat[1], 0.0])
            rh_norm   = np.linalg.norm(rep_horiz)
            if rh_norm > 0.1:
                rep_horiz /= rh_norm
                # Two candidate tangential directions (CCW and CW around obstacle)
                tang_ccw = np.array([-rep_horiz[1],  rep_horiz[0], 0.0])
                tang_cw  = np.array([ rep_horiz[1], -rep_horiz[0], 0.0])

                # Use horizontal velocity to pick the side the drone is already
                # drifting toward; if no horizontal velocity yet, use goal direction.
                v_h = np.array([vel_vec[0], vel_vec[1], 0.0])
                if np.linalg.norm(v_h) > 0.1:
                    selector = v_h
                elif ref_xyz is not None:
                    selector = np.array([ref_xyz[0] - drone_xyz[0],
                                         ref_xyz[1] - drone_xyz[1], 0.0])
                else:
                    selector = tang_ccw  # arbitrary default

                t_hat = tang_ccw if (tang_ccw @ selector >= tang_cw @ selector) \
                        else tang_cw
                mag_t = K_TANG * (1.0 / d_surf - 1.0 / r_act) / d_surf**2
                a_rep += mag_t * t_hat

    # Horizontal only — prevents z-component from fighting the altitude loop
    a_rep[2] = 0.0
    total = np.linalg.norm(a_rep)
    if total > MAX_ACC:
        a_rep *= MAX_ACC / total
    return a_rep


def compute_speed_scale(drone_xyz, vel_mag):
    """
    Speed scale factor in [SLOW_FACTOR, 1.0] evaluated at drone's actual position.
    1.0 = full speed (outside influence zone).  SLOW_FACTOR = closest approach.
    """
    r_act    = R_INFLUENCE + K_V * vel_mag
    max_prox = 0.0
    for obs in OBSTACLES:
        p_obs  = _cyl_to_xyz(obs['pos'])
        r_obs  = float(obs['radius'])
        d_surf = max(np.linalg.norm(drone_xyz - p_obs) - (r_obs + R_BASE), 0.0)
        if d_surf < (r_act - R_BASE):
            denom    = max(r_act - R_BASE, 1e-3)
            prox     = float(np.clip(1.0 - d_surf / denom, 0.0, 1.0))
            max_prox = max(max_prox, prox)
    return float(1.0 - (1.0 - SLOW_FACTOR) * max_prox**3)
