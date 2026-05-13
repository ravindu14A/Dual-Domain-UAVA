"""
drone.py — 6-DOF multirotor model supporting 4, 6, and 8 rotor configurations.

State vector (12 elements):
  [0:3]  position   (x, y, z)           world frame  [m]
  [3:6]  velocity   (vx, vy, vz)        world frame  [m/s]
  [6:9]  attitude   (φ roll, θ pitch, ψ yaw)         [rad]
  [9:12] body rates (p, q, r)           body frame   [rad/s]

Coordinate convention:
  World: x = East, y = North, z = Up  (right-hand)
  Body:  x = forward, y = left, z = up

Dynamics:
  Translational — Newton in world frame with thrust and gravity
  Rotational    — Euler's equations (simplified: no gyroscopic cross-terms)
  Euler rates   — standard kinematic relation from body rates

Power model:
  P_i = T_i^1.5 / sqrt(ρ · A_rotor · η_FM) / η_motor   (actuator-disk, per rotor)
  Battery drained at rate ΣP_i / η_battery
"""

import numpy as np
import scipy.constants as const


# ── Utility: rotation matrix (ZYX Euler, body → world) ───────────────────────

def rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """3×3 rotation matrix R such that v_world = R @ v_body."""
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)
    return np.array([
        [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [-st,    ct*sp,              ct*cp            ],
    ])


class DroneModel:
    """
    Rigid-body 6-DOF model of a symmetric multirotor (4 / 6 / 8 rotors).

    Individual rotor thrusts are the primary input; this class computes the
    resulting forces, moments, and battery draw.
    """

    SUPPORTED_ROTORS = (4, 6, 8)

    def __init__(self, config: dict, n_rotors: int,
                 initial_pos=(0.0, 0.0, 0.0)):
        if n_rotors not in self.SUPPORTED_ROTORS:
            raise ValueError(f"n_rotors must be one of {self.SUPPORTED_ROTORS}")

        self.config   = config
        self.n_rotors = n_rotors
        self.mass     = float(config["drone"]["weight"])
        self.g        = const.g

        # ── Propulsion parameters ─────────────────────────────────────────────
        self.rho       = float(config["air_density"])
        self.eta_fm    = float(config["hover_efficiency"])
        self.eta_m     = float(config["motor_efficiency"])
        self.k0        = float(config["profile_power_fraction"])
        self.max_T_mot = float(config["max_thrust_per_motor"])
        self.k_drag    = float(config.get("k_drag", 0.05))   # torque/thrust ratio

        # ── Rotor geometry ────────────────────────────────────────────────────
        self.r_rotor    = self._rotor_radius()
        self.A_rotor    = np.pi * self.r_rotor ** 2
        self.arm_length = self._arm_length()

        # ── Inertia ───────────────────────────────────────────────────────────
        inertia = config.get("inertia", {})
        if inertia:
            self.Ixx = float(inertia["Ixx"])
            self.Iyy = float(inertia["Iyy"])
            self.Izz = float(inertia["Izz"])
        else:
            L = self.arm_length
            m = self.mass
            # Treat drone as 4 point masses at ±L/√2 on each axis
            self.Ixx = 0.30 * m * L ** 2
            self.Iyy = 0.30 * m * L ** 2
            self.Izz = 0.55 * m * L ** 2

        # ── Body drag ─────────────────────────────────────────────────────────
        self.drag_coeff = float(config.get("body_drag_coeff", 2.5))

        # ── Rotor layout and mixer matrix ─────────────────────────────────────
        self.rotor_xy, self.rotor_spin = self._rotor_layout()
        self.A_mix     = self._build_mixer()          # (4, n_rotors)
        self.A_mix_inv = np.linalg.pinv(self.A_mix)  # (n_rotors, 4)

        # ── Battery ───────────────────────────────────────────────────────────
        bat_kg          = float(config["battery_weight_fraction"]) * self.mass
        self.bat_cap    = bat_kg * float(config["battery_specific_energy"])  # Wh
        self.bat_eff    = float(config["battery_efficiency"])
        self.battery    = self.bat_cap

        # ── State: [x,y,z, vx,vy,vz, φ,θ,ψ, p,q,r] ─────────────────────────
        self.state = np.zeros(12)
        self.state[0:3] = np.asarray(initial_pos, dtype=float)

        # Start with each rotor producing hover thrust
        self.rotor_thrusts = np.full(
            n_rotors, self.mass * self.g / n_rotors
        )

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _rotor_radius(self) -> float:
        """Per-rotor disk radius, keeping total disk area fixed for n ≥ 3."""
        r_ref = float(self.config["propeller_diameter"]) / 2.0
        n_ref = int(self.config["num_motors"])
        ref_area = n_ref * np.pi * r_ref ** 2
        if self.n_rotors <= 2:
            return float(np.sqrt(ref_area / (3.0 * np.pi)))
        return float(np.sqrt(ref_area / (self.n_rotors * np.pi)))

    def _arm_length(self) -> float:
        """
        Minimum arm length so adjacent propellers don't overlap, plus 10 % clearance.
        For n rotors equally spaced, adjacent centre-to-centre = 2·L·sin(π/n).
        """
        n = self.n_rotors
        if n == 4:
            # X-config: adjacent rotors at 90°, placed at 45° offset
            d_factor = np.sqrt(2.0)   # separation = L√2 for X-quad
        else:
            d_factor = 2.0 * np.sin(np.pi / n)
        return float(self.r_rotor * 2.0 / d_factor * 1.15)  # 15 % clearance

    def _rotor_layout(self):
        """
        Returns
        -------
        xy   : (n, 2)  body-frame (x_body, y_body) of each rotor centre [m]
        spin : (n,)    +1 = CW (positive yaw reaction), -1 = CCW
        """
        n = self.n_rotors
        L = self.arm_length

        if n == 4:
            # X-config: arms at 45°, 135°, 225°, 315° from body-x (forward)
            angles = [np.pi / 4 + np.pi / 2 * i for i in range(4)]
        else:
            # Symmetric equal-angle layout; 0° = forward (body-x)
            angles = [2.0 * np.pi * i / n for i in range(n)]

        xy   = np.array([[L * np.cos(a), L * np.sin(a)] for a in angles])
        spin = np.array([+1 if i % 2 == 0 else -1 for i in range(n)])
        return xy, spin

    def _build_mixer(self) -> np.ndarray:
        """
        Mixer matrix A (4 × n_rotors):
          row 0 : total thrust  T = Σ T_i
          row 1 : roll moment   τ_x = Σ T_i · y_i
          row 2 : pitch moment  τ_y = Σ T_i · (−x_i)
          row 3 : yaw moment    τ_z = Σ k_drag · spin_i · T_i

        Solve A @ T_rotors = [T, τ_x, τ_y, τ_z]  →  T_rotors = A⁺ @ cmd
        """
        A = np.zeros((4, self.n_rotors))
        A[0, :] = 1.0
        A[1, :] = self.rotor_xy[:, 1]            # y positions → roll
        A[2, :] = -self.rotor_xy[:, 0]            # -x positions → pitch
        A[3, :] = self.k_drag * self.rotor_spin   # drag torque → yaw
        return A

    # ── Actuator commands ─────────────────────────────────────────────────────

    def set_rotor_thrusts(self, thrusts: np.ndarray):
        """Clamp and store individual rotor thrust commands."""
        self.rotor_thrusts = np.clip(np.asarray(thrusts, dtype=float),
                                     0.0, self.max_T_mot)

    def mix(self, T_des: float,
            tau_x: float, tau_y: float, tau_z: float) -> np.ndarray:
        """
        Compute per-rotor thrusts from total thrust + moments (pseudo-inverse).
        Output is clamped to [0, max_T_per_motor].
        """
        cmd = np.array([T_des, tau_x, tau_y, tau_z], dtype=float)
        raw = self.A_mix_inv @ cmd
        return np.clip(raw, 0.0, self.max_T_mot)

    # ── Physics step ─────────────────────────────────────────────────────────

    def step(self, dt: float):
        """
        Integrate 6-DOF dynamics for one time step using
        the current rotor_thrusts.  Updates self.state in place.
        """
        phi, theta, psi = self.state[6:9]
        p, q, r         = self.state[9:12]
        v_world         = self.state[3:6]

        # ── Forces and moments from rotors ────────────────────────────────────
        T_total = float(np.dot(self.A_mix[0], self.rotor_thrusts))
        tau_x   = float(np.dot(self.A_mix[1], self.rotor_thrusts))
        tau_y   = float(np.dot(self.A_mix[2], self.rotor_thrusts))
        tau_z   = float(np.dot(self.A_mix[3], self.rotor_thrusts))

        # ── Translational dynamics (world frame) ──────────────────────────────
        R = rotation_matrix(phi, theta, psi)
        # Thrust in world frame (body z = [0,0,T] rotated to world)
        F_thrust = R @ np.array([0.0, 0.0, T_total])
        # Gravity
        F_grav = np.array([0.0, 0.0, -self.mass * self.g])
        # Linear drag opposing velocity
        F_drag = -self.drag_coeff * v_world

        a_world = (F_thrust + F_grav + F_drag) / self.mass

        # Semi-implicit Euler integration
        self.state[3:6] += a_world * dt
        self.state[0:3] += self.state[3:6] * dt

        # Ground constraint
        if self.state[2] < 0.0:
            self.state[2] = 0.0
            self.state[5] = max(self.state[5], 0.0)

        # ── Rotational dynamics (body frame) ──────────────────────────────────
        # Euler's equations (simplified: gyroscopic ω×Iω term omitted)
        alpha_x = tau_x / self.Ixx
        alpha_y = tau_y / self.Iyy
        alpha_z = tau_z / self.Izz

        p_new = p + alpha_x * dt
        q_new = q + alpha_y * dt
        r_new = r + alpha_z * dt

        self.state[9]  = p_new
        self.state[10] = q_new
        self.state[11] = r_new

        # ── Euler-angle kinematics ─────────────────────────────────────────────
        # φ̇ = p + (q·sinφ + r·cosφ)·tanθ
        # θ̇ = q·cosφ − r·sinφ
        # ψ̇ = (q·sinφ + r·cosφ) / cosθ
        sp, cp = np.sin(phi), np.cos(phi)
        ct = np.cos(theta)
        tt = np.tan(theta)
        ct = ct if abs(ct) > 1e-4 else np.sign(ct) * 1e-4  # singularity guard

        phi_dot   = p_new + (q_new * sp + r_new * cp) * tt
        theta_dot = q_new * cp - r_new * sp
        psi_dot   = (q_new * sp + r_new * cp) / ct

        self.state[6] += phi_dot   * dt
        self.state[7] += theta_dot * dt
        self.state[8] += psi_dot   * dt

    # ── Power and battery ─────────────────────────────────────────────────────

    def rotor_power(self) -> float:
        """Total electrical motor power [W] — actuator-disk model per rotor."""
        total = 0.0
        for T_i in self.rotor_thrusts:
            T_i = max(float(T_i), 0.01)
            P_i = T_i ** 1.5 / np.sqrt(self.rho * self.A_rotor * self.eta_fm)
            P_i *= (1.0 + self.k0)   # profile drag fraction
            total += P_i / self.eta_m
        return total

    def drain_battery(self, dt: float) -> float:
        """Drain battery by rotor_power() for dt seconds.  Returns Wh remaining."""
        P = self.rotor_power()
        self.battery = max(self.battery - P * dt / 3600.0 / self.bat_eff, 0.0)
        return self.battery

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        return self.state[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6].copy()

    @property
    def attitude(self) -> np.ndarray:
        """Roll, pitch, yaw [rad]."""
        return self.state[6:9].copy()

    @property
    def body_rates(self) -> np.ndarray:
        return self.state[9:12].copy()

    @property
    def hover_thrust(self) -> float:
        return self.mass * self.g

    @property
    def rotation(self) -> np.ndarray:
        phi, theta, psi = self.state[6:9]
        return rotation_matrix(phi, theta, psi)
