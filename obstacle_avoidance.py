# obstacle_avoidance.py
# Potential-field obstacle avoidance layer for quadcopterSC.
#
# Computes a Cartesian repulsive deflection that is added to the reference
# position each sim step. Works with both cylindrical and Cartesian reference
# frames — coordinate conversion is handled inside quadcopterSC.
#
# Tuning guide
# ────────────
# R_BASE       : hard minimum clearance from obstacle surface (m). Set this to
#                at least rotor half-span + structural margin.
# K_V          : velocity-scaling factor (s). Activation radius grows by
#                K_V * |v|, so faster flight triggers avoidance earlier.
# K_REP        : radial repulsive gain (pushes reference away from obstacle).
# R_INFLUENCE  : base distance from obstacle surface where repulsion starts (m).
#                At speed v the effective activation radius = R_INFLUENCE + K_V*v.
# MAX_DEFL     : hard cap on deflection magnitude (m). Prevents runaway near
#                very close obstacles.
# SMOOTH_TAU   : first-order low-pass time constant (s) on the deflection
#                output. Prevents sudden reference jumps reaching the PID.

import numpy as np 
# ──────────────────────────────────────────────────────────────────────────────
#  OBSTACLE DEFINITIONS  (cylindrical coordinates)
#  Each entry: {'pos': [r_m, theta_deg, z_m], 'radius': float}
#    r_m       – radial distance from tower axis [m]
#    theta_deg – azimuth angle [deg], 0 = +x direction
#    z_m       – height above sea level [m]  (z=0 convention same as SC sim)
#    radius    – obstacle sphere radius [m]
# ──────────────────────────────────────────────────────────────────────────────
OBSTACLES = [
     #{'pos': [6, 2, 20], 'radius': 0.3},
]

# ──────────────────────────────────────────────────────────────────────────────
#  TUNING PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
R_BASE      = 1.5   # [m]   min clearance — shown as orange safety shell in 3D plot
K_V         = 0.35  # [s]   velocity scaling: r_act = R_INFLUENCE + K_V * |v|
K_REP       = 15.0  # [—]   radial repulsive gain
R_INFLUENCE = 6.0   # [m]   base activation distance from obstacle surface
MAX_DEFL    = 3.0   # [m]   hard cap on reference shift
SMOOTH_TAU  = 0.05  # [s]   fast decay — clears cleanly once obstacle is passed
SLOW_FACTOR = 0.35  # [—]   min speed fraction near obstacle (35% of nominal)

# ──────────────────────────────────────────────────────────────────────────────
#  INTERNAL HELPER
# ──────────────────────────────────────────────────────────────────────────────
def _cyl_to_xyz(pos_cyl):
    """Convert cylindrical [r, theta_deg, z] → Cartesian [x, y, z]."""
    r, th, z = float(pos_cyl[0]), float(pos_cyl[1]), float(pos_cyl[2])
    th_rad = np.radians(th)
    return np.array([r * np.cos(th_rad), r * np.sin(th_rad), z])

# ──────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────
def compute_raw_deflection(ref_xyz, vel_vec):
    """
    Return the instantaneous Cartesian repulsive deflection vector.

    ref_xyz : (3,) reference position in Cartesian [m]
    vel_vec : (3,) current velocity vector [m/s] — magnitude scales activation radius

    Returns (3,) deflection [dx, dy, dz].
    """
    vel_vec = np.asarray(vel_vec, dtype=float)
    vel_mag = np.linalg.norm(vel_vec)
    vel_hat = vel_vec / (vel_mag + 1e-9)

    defl  = np.zeros(3)
    r_act = R_INFLUENCE + K_V * vel_mag   # velocity-scaled activation distance
    for obs in OBSTACLES:
        p_obs = _cyl_to_xyz(obs['pos'])
        r_obs = float(obs['radius'])

        diff = ref_xyz - p_obs
        d    = np.linalg.norm(diff)
        if d < 1e-6:
            diff = np.array([1., 0., 0.])
            d    = 1e-6

        # Measure distance from the safety surface (physical radius + R_BASE clearance).
        # This means maximum force is reached at R_BASE from the hull, not at the hull.
        d_surf  = max(d - (r_obs + R_BASE), 1e-3)
        rad_hat = diff / d

        if d_surf < r_act:
            base_mag = K_REP * (1.0 / d_surf - 1.0 / r_act) / d_surf**2

            # For head-on approach (radial force opposes motion): redirect the
            # push to the obstacle's outward horizontal bearing from the tower.
            # A backward-along-trajectory push stalls the reference; a sideways
            # push steers it cleanly around without creating orbital dynamics.
            if vel_mag > 0.1 and (rad_hat @ vel_hat) < -0.5:
                obs_xy      = np.array([p_obs[0], p_obs[1], 0.])
                obs_xy_norm = np.linalg.norm(obs_xy)
                eff_dir = (obs_xy / obs_xy_norm) if obs_xy_norm > 0.1 \
                          else np.array([1., 0., 0.])
            else:
                eff_dir = rad_hat

            defl += base_mag * eff_dir

    # Force avoidance to be purely horizontal — prevents the z-component from
    # fighting the descent and causing the deflection vector to rotate as the
    # drone descends past the obstacle (which produces zigzag oscillation).
    defl[2] = 0.0

    total = np.linalg.norm(defl)
    if total > MAX_DEFL:
        defl *= MAX_DEFL / total
    return defl


def compute_speed_scale(ref_xyz, vel_mag):
    """
    Return a speed scale factor in [SLOW_FACTOR, 1.0].

    1.0         = full speed (outside influence zone)
    SLOW_FACTOR = slowest (at obstacle surface + R_BASE)

    The scale falls off smoothly (cubic) from 1.0 at r_act to SLOW_FACTOR at R_BASE,
    so the drone decelerates gradually as it approaches and re-accelerates once clear.
    """
    r_act    = R_INFLUENCE + K_V * vel_mag
    max_prox = 0.0                          # 0 = far, 1 = at or inside R_BASE
    for obs in OBSTACLES:
        p_obs  = _cyl_to_xyz(obs['pos'])
        r_obs  = float(obs['radius'])
        
        # Modified to match the strict safety hull calculation of the deflection function
        d_surf = max(np.linalg.norm(ref_xyz - p_obs) - (r_obs + R_BASE), 0.0)
        
        if d_surf < (r_act - R_BASE):
            # linear proximity: 0 at r_act boundary, 1 at R_BASE boundary
            denominator = max((r_act - R_BASE), 1e-3)
            prox = 1.0 - (d_surf / denominator)
            prox = float(np.clip(prox, 0.0, 1.0))
            max_prox = max(max_prox, prox)
            
    # cubic ease-in so slowdown is gentle far out, stronger close in
    scale = 1.0 - (1.0 - SLOW_FACTOR) * max_prox**3
    return float(scale)