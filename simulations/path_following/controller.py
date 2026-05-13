"""
controller.py — Cascaded PID path-following controller.

Architecture
------------
Outer loop  (50 Hz):  position error  →  desired linear acceleration
                       desired accel   →  desired attitude (φ, θ) + total thrust

Inner loop  (50 Hz):  attitude error   →  body-frame moments (τ_x, τ_y, τ_z)

Mixer       (drone.py): [T, τ_x, τ_y, τ_z] →  individual rotor thrusts

Coordinate convention matches drone.py:
  Body x = forward, y = left, z = up.
  Positive φ (roll)  = right wing down  →  force to the RIGHT
  Positive θ (pitch) = nose up           →  force FORWARD
"""

import numpy as np


class _PID:
    def __init__(self, Kp: float, Ki: float, Kd: float,
                 out_min: float = -np.inf, out_max: float = np.inf,
                 windup: float = 100.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.out_min, self.out_max = out_min, out_max
        self.windup = windup
        self._integral   = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        self._integral = float(np.clip(
            self._integral + error * dt,
            -self.windup, self.windup))
        deriv = (error - self._prev_error) / max(dt, 1e-9)
        self._prev_error = error
        raw = self.Kp * error + self.Ki * self._integral + self.Kd * deriv
        return float(np.clip(raw, self.out_min, self.out_max))


class PathFollowController:
    """
    Cascaded PID controller that drives a DroneModel to track a moving
    3-D target position at each time step.

    Usage:
        ctrl = PathFollowController(config, drone)
        ...
        thrusts = ctrl.compute(target_pos, drone.state, dt)
        drone.set_rotor_thrusts(thrusts)
        drone.step(dt)
    """

    def __init__(self, config: dict, drone):
        self.drone    = drone
        self.g        = 9.80665
        self.mass     = drone.mass
        self.max_T    = drone.max_T_mot * drone.n_rotors

        fc  = config.get("flight_controller", {})
        pp  = fc.get("pos_pid",  {})
        ap  = fc.get("att_pid",  {})

        a_max = float(pp.get("max_accel", 6.0))

        # ── Position PIDs  (output: desired acceleration [m/s²]) ──────────────
        self.pid_x = _PID(float(pp.get("Kp",   2.5)),
                          float(pp.get("Ki",   0.04)),
                          float(pp.get("Kd",   4.5)),
                          out_min=-a_max, out_max=a_max)
        self.pid_y = _PID(float(pp.get("Kp",   2.5)),
                          float(pp.get("Ki",   0.04)),
                          float(pp.get("Kd",   4.5)),
                          out_min=-a_max, out_max=a_max)
        self.pid_z = _PID(float(pp.get("Kp_z", 3.5)),
                          float(pp.get("Ki_z", 0.08)),
                          float(pp.get("Kd_z", 6.0)),
                          out_min=-a_max, out_max=a_max)

        # ── Attitude PIDs  (output: body-frame moment [N·m]) ──────────────────
        tau_max = drone.max_T_mot * drone.arm_length * drone.n_rotors
        self.pid_phi   = _PID(float(ap.get("Kp",     50.0)),
                              float(ap.get("Ki",      0.3)),
                              float(ap.get("Kd",     12.0)),
                              out_min=-tau_max, out_max=tau_max)
        self.pid_theta = _PID(float(ap.get("Kp",     50.0)),
                              float(ap.get("Ki",      0.3)),
                              float(ap.get("Kd",     12.0)),
                              out_min=-tau_max, out_max=tau_max)
        self.pid_psi   = _PID(float(ap.get("Kp_psi",  8.0)),
                              float(ap.get("Ki_psi",  0.05)),
                              float(ap.get("Kd_psi",  3.0)),
                              out_min=-tau_max * 0.5, out_max=tau_max * 0.5)

        self.max_tilt     = float(fc.get("max_tilt_deg", 30.0)) * np.pi / 180.0
        self.track_yaw    = bool(fc.get("track_yaw", False))
        self.desired_yaw  = 0.0     # fixed yaw setpoint when track_yaw=False

    def reset(self):
        for pid in (self.pid_x, self.pid_y, self.pid_z,
                    self.pid_phi, self.pid_theta, self.pid_psi):
            pid.reset()

    def set_yaw_target(self, psi: float):
        self.desired_yaw = float(psi)

    # ── Main compute call ─────────────────────────────────────────────────────

    def compute(self, target_pos: np.ndarray,
                drone_state: np.ndarray,
                dt: float) -> np.ndarray:
        """
        Parameters
        ----------
        target_pos   : (3,) desired world-frame position [m]
        drone_state  : (12,) current drone state vector
        dt           : integration timestep [s]

        Returns
        -------
        thrusts : (n_rotors,) individual rotor thrust commands [N]
        """
        pos         = drone_state[0:3]
        phi, theta, psi = drone_state[6:9]

        # ── Outer loop: position error → desired acceleration ─────────────────
        err = target_pos - pos
        ax_des = self.pid_x.update(err[0], dt)
        ay_des = self.pid_y.update(err[1], dt)
        az_des = self.pid_z.update(err[2], dt)

        # ── Desired thrust magnitude (gravity feed-forward) ───────────────────
        Fx = self.mass * ax_des
        Fy = self.mass * ay_des
        Fz = self.mass * (az_des + self.g)

        T_des = float(np.sqrt(Fx**2 + Fy**2 + Fz**2))
        T_des = float(np.clip(T_des, self.mass * self.g * 0.3, self.max_T))

        # ── Desired attitude from desired force vector ─────────────────────────
        # Rotate desired force direction into yaw-aligned body frame
        cy, sy = np.cos(psi), np.sin(psi)
        Fx_b =  Fx * cy + Fy * sy   # forward component in body x
        Fy_b = -Fx * sy + Fy * cy   # lateral component in body y
        Fz_b = Fz                    # vertical (common denominator)

        # Desired pitch: nose up to produce forward thrust
        # Desired roll:  roll left to produce leftward thrust  (φ < 0 → force +y)
        theta_des = float(np.arctan2(Fx_b, Fz_b))
        phi_des   = float(np.arctan2(-Fy_b, Fz_b))

        theta_des = float(np.clip(theta_des, -self.max_tilt, self.max_tilt))
        phi_des   = float(np.clip(phi_des,   -self.max_tilt, self.max_tilt))

        # Yaw setpoint
        if self.track_yaw and np.linalg.norm(err[:2]) > 0.5:
            self.desired_yaw = float(np.arctan2(err[1], err[0]))
        psi_des = self.desired_yaw

        # Normalise yaw error to (−π, π)
        psi_err = (psi_des - psi + np.pi) % (2 * np.pi) - np.pi

        # ── Inner loop: attitude error → moments ──────────────────────────────
        tau_x = self.pid_phi.update(phi_des   - phi,   dt)
        tau_y = self.pid_theta.update(theta_des - theta, dt)
        tau_z = self.pid_psi.update(psi_err,             dt)

        # ── Rotor mixer ───────────────────────────────────────────────────────
        return self.drone.mix(T_des, tau_x, tau_y, tau_z)
