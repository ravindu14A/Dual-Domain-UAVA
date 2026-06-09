# mpc_planner.py
"""
Receding-horizon MPC planner for obstacle avoidance.

Architecture
────────────
                      ┌──────────────────┐
  offline ref_pos/vel │                  │  rp, rv, k_ref
  ───────────────────►│   MPCPlanner     ├───────────────► PID controller
  drone pos / vel     │                  │
  ───────────────────►│                  │
                      └──────────────────┘

Normal mode (no obstacle in horizon):
    Passes ref_pos[:, k_ref] and ref_vel[:, k_ref] through unchanged.

Obstacle detected:
    Solves a receding-horizon trajectory optimisation over a coarse planning
    grid (dt_mpc >> dt_sim), interpolates the result back to the fine sim
    timestep, and serves that as the active reference.  Simultaneously the
    offline k_ref is frozen (the drone is off the planned path, so the
    event-triggered waypoint manager would stall it anyway).

After the obstacle is clear:
    Finds the closest waypoint on the offline path that is beyond the
    obstacle and within reach, then jumps k_ref there so the waypoint
    manager resumes cleanly without trying to re-visit waypoints the drone
    already flew past.

Model
─────
3-D double integrator  (position + velocity state, acceleration input).
The inner PID attitude loop is fast relative to the MPC horizon, so treating
the outer loop as an acceleration-controlled point mass is a good planning
approximation.

    X = [x, y, z, vx, vy, vz]
    U = [ax, ay, az]

    X[k+1] = A @ X[k] + B @ U[k]

Waypoint density note
─────────────────────
The offline trajectory is generated at dt_sim = 0.005 s.  At a climb speed
of ~2 m/s that is ~100 waypoints/m — far too dense to use directly for
prediction.  The obstacle-detection scan looks detect_steps waypoints ahead
(default 2000 = 10 s / 20 m) to trigger early enough.  The MPC itself
plans on a coarser grid (dt_mpc = 0.15 s, N_pred = 40 → 6 s / ~12 m
planning horizon) and the state trajectory is linearly interpolated back
to the fine sim timestep before being handed to the PID.
"""

import numpy as np
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────────────────────
#  TUNING PARAMETERS  — weights are intentionally separated so they can be
#  changed here without touching the class.  We'll tune Q/R together.
# ─────────────────────────────────────────────────────────────────────────────

# Prediction horizon
N_PRED   = 40      # MPC steps (at dt_mpc each)
DT_MPC   = 0.15    # [s] planning sample period  (fine dt = 0.005 s)
                   # → fine_per_mpc = DT_MPC / 0.005 = 30 fine steps per MPC step
                   # → horizon = N_PRED * DT_MPC = 6 s

# Detection
DETECT_STEPS = 2000   # fine-grid steps to scan ahead for obstacles
                      # at dt=0.005 s → 10 s / ~20 m look-ahead at 2 m/s

# Re-solve rate (receding horizon)
SOLVE_EVERY = 30      # re-solve every N fine steps = every DT_MPC seconds

# Safety clearance
R_SAFETY = 2.0        # [m]  extra margin beyond obstacle radius

# Control bounds
U_MAX = 6.0           # [m/s²]  max acceleration in any axis

# ── Cost matrices ─────────────────────────────────────────────────────────────
# Q_POS  : position tracking.  z weighted 2× (vertical climb must continue).
# Q_VEL  : velocity tracking (10× softer than position).
# Q_TERM : terminal position cost (10× Q_POS) — strong return to planned path
#          at the end of the horizon so the drone doesn't wander indefinitely.
# R_CTRL : control effort.  Lateral axes cheaper than vertical so the planner
#          prefers to steer sideways rather than fight the climb.
# W_OBS  : soft obstacle penalty weight.  Penalty = W_OBS * max(0, d_viol)²
#          where d_viol = R_SAFETY − (dist_to_obs_surface).
#          At contact (d_viol = R_SAFETY = 2 m): cost = W_OBS * 4.
#          Must dominate the position-tracking term to guarantee avoidance.

Q_POS  = np.diag([1.0,  1.0,  2.0 ])   # position  [x, y, z]
Q_VEL  = np.diag([0.1,  0.1,  0.2 ])   # velocity  [vx, vy, vz]
Q_TERM = np.diag([10.0, 10.0, 20.0])   # terminal  position
R_CTRL = np.diag([0.05, 0.05, 0.10])   # control   [ax, ay, az]
W_OBS  = 500.0                          # obstacle penalty


# ─────────────────────────────────────────────────────────────────────────────
class MPCPlanner:
    """
    Receding-horizon MPC planner.  Instantiate once before the sim loop and
    call .query() every fine timestep inside the loop.

    Parameters
    ----------
    obstacles   : list[dict]  {'pos': [r_m, theta_deg, z_m], 'radius': float}
    cyl_to_xyz  : callable  [r, theta_deg, z] → np.ndarray([x, y, z])
    use_cyl_ref : bool  True if ref_pos / ref_vel are in cylindrical (r, θ, z)
    dt_sim      : float  fine simulation timestep [s]  (default 0.005)
    """

    def __init__(
        self,
        obstacles,
        cyl_to_xyz,
        use_cyl_ref = True,
        dt_sim      = 0.005,
        N_pred      = N_PRED,
        dt_mpc      = DT_MPC,
        detect_steps= DETECT_STEPS,
        solve_every = SOLVE_EVERY,
        R_safety    = R_SAFETY,
        u_max       = U_MAX,
        Q_pos       = Q_POS,
        Q_vel       = Q_VEL,
        Q_term      = Q_TERM,
        R_ctrl      = R_CTRL,
        W_obs       = W_OBS,
    ):
        self.obs          = obstacles
        self._c2xyz       = cyl_to_xyz
        self.cyl          = use_cyl_ref
        self.dt_sim       = dt_sim
        self.N            = N_pred
        self.dt           = dt_mpc
        self.det_steps    = detect_steps
        self.solve_every  = solve_every
        self.R_saf        = R_safety
        self.u_max        = u_max
        self.Q_p          = Q_pos.copy()
        self.Q_v          = Q_vel.copy()
        self.Q_t          = Q_term.copy()
        self.R            = R_ctrl.copy()
        self.W_obs        = W_obs

        # fine steps per MPC step (must be integer)
        self._fpm = max(1, round(dt_mpc / dt_sim))

        # Double integrator matrices at dt_mpc
        dt = dt_mpc
        self._A = np.block([
            [np.eye(3),      dt * np.eye(3)        ],
            [np.zeros((3,3)), np.eye(3)             ],
        ])
        self._B = np.block([
            [0.5 * dt**2 * np.eye(3)],
            [dt          * np.eye(3)],
        ])

        # Runtime state
        self._active    = False    # True: MPC is overriding the offline ref
        self._traj      = None     # planned state trajectory (N+1, 6) Cartesian
        self._fine_ctr  = 0        # fine steps elapsed since last solve
        self._k_frozen  = 0        # k_ref at the moment MPC activated

    # ── coordinate helpers ───────────────────────────────────────────────────

    def _obs_centers(self):
        return [self._c2xyz(o['pos']) for o in self.obs]

    def _rp_to_xyz(self, rp):
        """Reference position (cyl or cart) → Cartesian (3,)."""
        if self.cyl:
            r, th, z = float(rp[0]), float(rp[1]), float(rp[2])
            return np.array([r * np.cos(th), r * np.sin(th), z])
        return np.asarray(rp[:3], dtype=float)

    def _rv_to_xyz(self, rv, rp):
        """Reference velocity (cyl or cart) → Cartesian (3,)."""
        if not self.cyl:
            return np.asarray(rv[:3], dtype=float)
        r, th = max(float(rp[0]), 1e-6), float(rp[1])
        vr, vth, vz = float(rv[0]), float(rv[1]), float(rv[2])
        cs, sn = np.cos(th), np.sin(th)
        return np.array([vr*cs - r*vth*sn, vr*sn + r*vth*cs, vz])

    def _xyz_to_rp(self, xyz):
        """Cartesian (3,) → reference position in sim frame."""
        if not self.cyl:
            return xyz.copy()
        x, y, z = xyz
        return np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x), z])

    def _xyz_to_rv(self, vel_xyz, rp_cyl):
        """Cartesian velocity → reference velocity in sim frame."""
        if not self.cyl:
            return vel_xyz.copy()
        r  = max(float(rp_cyl[0]), 1e-6)
        th = float(rp_cyl[1])
        vx, vy, vz = vel_xyz
        cs, sn = np.cos(th), np.sin(th)
        return np.array([vx*cs + vy*sn, (-vx*sn + vy*cs) / r, vz])

    # ── obstacle utilities ───────────────────────────────────────────────────

    def _min_surf_dist(self, pos_xyz):
        """Minimum distance from pos_xyz to any obstacle surface."""
        d = np.inf
        for o, c in zip(self.obs, self._obs_centers()):
            d = min(d, np.linalg.norm(pos_xyz - c) - float(o['radius']))
        return d

    def _collision_ahead(self, k_ref, ref_pos):
        """
        Scan detect_steps fine-grid waypoints ahead.
        Returns True if any waypoint encroaches on the safety radius.
        """
        n_wp = ref_pos.shape[1]
        centers = self._obs_centers()
        for k in range(k_ref, min(k_ref + self.det_steps, n_wp)):
            p = self._rp_to_xyz(ref_pos[:, k])
            for o, c in zip(self.obs, centers):
                if np.linalg.norm(p - c) < float(o['radius']) + self.R_saf:
                    return True
        return False

    # ── MPC internals ────────────────────────────────────────────────────────

    def _sample_offline_ref(self, k_ref, ref_pos, ref_vel):
        """
        Sample the dense offline reference at MPC resolution.
        Returns ref_xyz (N+1, 3) and ref_vxyz (N+1, 3) in Cartesian.
        """
        n_wp = ref_pos.shape[1]
        ref_xyz  = np.zeros((self.N + 1, 3))
        ref_vxyz = np.zeros((self.N + 1, 3))
        for i in range(self.N + 1):
            ki = min(k_ref + i * self._fpm, n_wp - 1)
            ref_xyz[i]  = self._rp_to_xyz(ref_pos[:, ki])
            ref_vxyz[i] = self._rv_to_xyz(ref_vel[:, ki], ref_pos[:, ki])
        return ref_xyz, ref_vxyz

    def _propagate(self, x0, U):
        """Propagate double-integrator model. x0 (6,), U (N,3) → X (N+1,6)."""
        X = np.empty((self.N + 1, 6))
        X[0] = x0
        for k in range(self.N):
            X[k + 1] = self._A @ X[k] + self._B @ U[k]
        return X

    def _cost(self, U_flat, x0, ref_xyz, ref_vxyz):
        """Objective: tracking + control effort + obstacle soft penalty."""
        U = U_flat.reshape(self.N, 3)
        X = self._propagate(x0, U)
        centers = self._obs_centers()
        J = 0.0
        for k in range(self.N):
            dp = X[k, :3] - ref_xyz[k]
            dv = X[k, 3:] - ref_vxyz[k]
            J += dp @ self.Q_p @ dp
            J += dv @ self.Q_v @ dv
            J += U[k] @ self.R @ U[k]
            # Soft obstacle penalty — quadratic growth inside safety radius
            for o, c in zip(self.obs, centers):
                d_surf = np.linalg.norm(X[k, :3] - c) - float(o['radius'])
                viol   = self.R_saf - d_surf          # > 0 inside safety zone
                if viol > 0:
                    J += self.W_obs * viol**2
        # Terminal cost — strong pull back to planned path at horizon end
        dp_t = X[self.N, :3] - ref_xyz[self.N]
        J += dp_t @ self.Q_t @ dp_t
        return J

    def _solve(self, drone_xyz, drone_vel, k_ref, ref_pos, ref_vel):
        """
        Solve MPC.  Returns planned state trajectory (N+1, 6) in Cartesian.
        Uses L-BFGS-B (handles bounds, fast for ~120 variables).
        """
        x0       = np.concatenate([drone_xyz, drone_vel])
        ref_xyz, ref_vxyz = self._sample_offline_ref(k_ref, ref_pos, ref_vel)
        U0       = np.zeros(self.N * 3)
        bounds   = [(-self.u_max, self.u_max)] * (self.N * 3)
        res = minimize(
            self._cost, U0,
            args    = (x0, ref_xyz, ref_vxyz),
            method  = 'L-BFGS-B',
            bounds  = bounds,
            options = {'maxiter': 300, 'ftol': 1e-5, 'gtol': 1e-4},
        )
        U_opt = res.x.reshape(self.N, 3)
        return self._propagate(x0, U_opt)

    # ── resume waypoint ──────────────────────────────────────────────────────

    def _find_resume_kref(self, drone_xyz, k_ref, ref_pos):
        """
        After avoidance: scan forward past the obstacle zone and return the
        index of the closest clear waypoint ahead of the drone.

        Strategy:
          1. Skip any waypoints still inside the safety radius (obstacle zone).
          2. Among the clear waypoints found, pick the one closest to the
             drone's current position.
          3. Stop searching once distance starts growing again (we passed the
             minimum) to avoid picking a distant future waypoint.
        """
        n_wp    = ref_pos.shape[1]
        centers = self._obs_centers()
        radii   = [float(o['radius']) + self.R_saf for o in self.obs]

        best_k    = min(k_ref + self.det_steps, n_wp - 1)
        best_dist = np.inf
        in_clear  = False

        search_end = min(k_ref + self.det_steps * 2, n_wp)
        for k in range(k_ref, search_end):
            p = self._rp_to_xyz(ref_pos[:, k])
            # Check if this waypoint is clear of all obstacles
            clear = all(
                np.linalg.norm(p - c) >= r
                for c, r in zip(centers, radii)
            )
            if not clear:
                in_clear = False
                continue
            # First clear waypoint signals we are past the obstacle
            if not in_clear:
                in_clear = True
            d = np.linalg.norm(drone_xyz - p)
            if d < best_dist:
                best_dist = d
                best_k    = k
            elif in_clear and d > best_dist + 3.0:
                # Distance increasing — we found the closest clear point
                break

        return best_k

    # ── interpolation ────────────────────────────────────────────────────────

    def _interp_traj(self, fine_step):
        """
        Map fine_step (since last solve) → interpolated Cartesian (pos, vel).
        Linear interpolation between MPC state samples.
        """
        mpc_k = fine_step // self._fpm
        frac  = (fine_step % self._fpm) / self._fpm
        mpc_k = min(mpc_k, self.N - 1)
        pos = (1.0 - frac) * self._traj[mpc_k,   :3] + \
                     frac  * self._traj[mpc_k + 1, :3]
        vel = (1.0 - frac) * self._traj[mpc_k,   3:] + \
                     frac  * self._traj[mpc_k + 1, 3:]
        return pos, vel

    # ── main interface ────────────────────────────────────────────────────────

    def query(self, k_ref, ref_pos, ref_vel, drone_pos_xyz, drone_vel_xyz):
        """
        Call once per fine simulation timestep inside the sim loop.

        Parameters
        ----------
        k_ref         : int    current offline waypoint index
        ref_pos       : (3, N_sim)  offline reference positions
        ref_vel       : (3, N_sim)  offline reference velocities
        drone_pos_xyz : (3,)   drone Cartesian position  [m]
        drone_vel_xyz : (3,)   drone Cartesian velocity  [m/s]

        Returns
        -------
        rp        : (3,) reference position in sim's native frame (cyl or cart)
        rv        : (3,) reference velocity in sim's native frame
        k_ref_out : int  updated k_ref (may jump forward after avoidance)
        active    : bool True = MPC is overriding the offline reference
        """

        # ── INACTIVE ─────────────────────────────────────────────────────────
        if not self._active:
            if self._collision_ahead(k_ref, ref_pos):
                # Trigger: solve initial trajectory and activate
                print(f"[MPC] Obstacle in look-ahead — activating at k_ref={k_ref}")
                self._traj     = self._solve(drone_pos_xyz, drone_vel_xyz,
                                              k_ref, ref_pos, ref_vel)
                self._fine_ctr = 0
                self._k_frozen = k_ref
                self._active   = True
            else:
                # Pass-through: return offline reference unchanged
                return (ref_pos[:, k_ref].copy(),
                        ref_vel[:, k_ref].copy(),
                        k_ref, False)

        # ── ACTIVE ───────────────────────────────────────────────────────────
        self._fine_ctr += 1

        # Receding-horizon re-solve
        if self._fine_ctr % self.solve_every == 0:
            # Advance the offline anchor by how many fine steps we have taken
            k_anchor = min(self._k_frozen + self._fine_ctr, ref_pos.shape[1] - 1)
            self._traj     = self._solve(drone_pos_xyz, drone_vel_xyz,
                                          k_anchor, ref_pos, ref_vel)
            self._fine_ctr = 0   # reset counter; new trajectory starts at step 0

        # Interpolate current fine step on the planned trajectory
        pos_xyz, vel_xyz = self._interp_traj(self._fine_ctr)

        # ── Deactivation check ───────────────────────────────────────────────
        horizon_elapsed = self._fine_ctr >= self.N * self._fpm
        obstacle_clear  = self._min_surf_dist(drone_pos_xyz) > self.R_saf * 1.5

        if horizon_elapsed or obstacle_clear:
            k_resume = self._find_resume_kref(drone_pos_xyz, self._k_frozen, ref_pos)
            self._active   = False
            self._traj     = None
            self._fine_ctr = 0
            print(f"[MPC] Obstacle cleared — resuming offline path at k_ref={k_resume}")
            return (ref_pos[:, k_resume].copy(),
                    ref_vel[:, k_resume].copy(),
                    k_resume, False)

        # Convert MPC Cartesian output back to sim reference frame
        rp = self._xyz_to_rp(pos_xyz)
        rv = self._xyz_to_rv(vel_xyz, rp)
        return rp, rv, k_ref, True


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION NOTES FOR quadcopterSC.py
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Add config flag near USE_EKF / OA_MODE:
#
#    USE_MPC = True   # True: MPC planner sits between offline ref and PID
#                     # False: raw ref_pos[:, k_ref] used directly
#
# 2. Import and instantiate once before the sim loop
#    (after ref_pos / ref_vel are built, after obstacles are loaded):
#
#    if USE_MPC and OA_MODE != "none":
#        from mpc_planner import MPCPlanner
#        _mpc = MPCPlanner(
#            obstacles   = _obs_list,
#            cyl_to_xyz  = _obs_cyl2xyz,
#            use_cyl_ref = _USE_CYL_REF,
#            dt_sim      = dt,
#        )
#
# 3. Inside the sim loop, replace the raw ref lookup:
#
#    # was:
#    _rp = ref_pos[:, k_ref].copy()
#    _rv = ref_vel[:, k_ref]
#
#    # becomes:
#    if USE_MPC and OA_MODE != "none":
#        _rp, _rv, k_ref, _mpc_active = _mpc.query(
#            k_ref, ref_pos, ref_vel, pos, vel)
#    else:
#        _rp = ref_pos[:, k_ref].copy()
#        _rv = ref_vel[:, k_ref]
#
#    _ra = ref_acc[:, k_ref]   # feedforward acc (still from offline ref)
#    _ry = ref_yaw[  k_ref]
#
# 4. The event-triggered k_ref advance (USE_EVENT_TRIG block) runs after
#    the MPC query.  When MPC is active the drone is off the planned path so
#    event triggering won't fire — k_ref stays frozen naturally.  When MPC
#    deactivates it returns the jump-forward k_resume, so the waypoint
#    manager picks up from the correct point on the offline path.
#
# ─────────────────────────────────────────────────────────────────────────────
