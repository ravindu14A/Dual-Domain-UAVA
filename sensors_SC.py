"""
sensors_SC.py  --  Discrete sensor models for the aerial phase (quadcopterSC)
DSE Team 30 | UAUV for Offshore Wind Turbine Inspection

Sensors modelled:
  IMUSensor     -- 9-DOF: accelerometer + gyroscope + magnetometer
  BaroSensor    -- barometric altimeter
  GNSSSensor    -- GNSS position + velocity
  LiDAR2DSensor -- horizontal scan, extracts standoff distance and bearing to turbine centre

All sensors are discrete: each has an update_rate [Hz]. Call tick() every simulation
step; it returns a fresh measurement at the correct interval and holds the last
measurement otherwise.  True states are connected here; the Kalman filter sits above.

Convention: ENU inertial frame, body x=forward y=left z=up (FLU), ZYX Euler.
"""

import numpy as np


# ── rotation helper (same convention as quadcopterSC) ────────────────────────

def _rot_ZYX(phi, theta, psi):
    """Body-to-inertial (R).  R.T maps inertial → body."""
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)
    return np.array([
        [cy*ct,        cy*st*sp - sy*cp,   cy*st*cp + sy*sp],
        [sy*ct,        sy*st*sp + cy*cp,   sy*st*cp - cy*sp],
        [  -st,               ct*sp,              ct*cp     ],
    ])


# ── base class ────────────────────────────────────────────────────────────────

class _DiscreteSensor:
    """Holds update rate and last-fire logic. Subclasses implement _measure()."""

    update_rate: float = 1.0   # [Hz] override in subclass

    def __init__(self):
        self._last_tick = -1       # integer tick counter at last fire
        self._cache     = None     # last measurement (held between updates)

    def _step_index(self, t, dt):
        """Simulation step index corresponding to time t."""
        return int(round(t / dt))

    def _due(self, t, dt):
        """True if the sensor should fire this simulation step."""
        sim_hz      = 1.0 / dt
        every_n     = max(1, round(sim_hz / self.update_rate))
        step        = self._step_index(t, dt)
        return step % every_n == 0

    def tick(self, t, dt, *args, **kwargs):
        """
        Call every simulation step.
        Returns a fresh measurement when the sensor fires, None otherwise.
        """
        if self._due(t, dt):
            self._cache = self._measure(*args, **kwargs)
            return self._cache
        return None


# ── IMU ───────────────────────────────────────────────────────────────────────

class IMUSensor(_DiscreteSensor):
    """
    9-DOF IMU: accelerometer, gyroscope, magnetometer.

    update_rate : 200 Hz

    Inputs (true values from plant state):
      accel_inertial (3,) -- true inertial acceleration  [m/s²]
      euler          (3,) -- [phi, theta, psi]            [rad]
      wb             (3,) -- body angular rates [p, q, r] [rad/s]

    Outputs (body frame):
      accel_body (3,) -- specific force = (a_inertial - g) rotated to body [m/s²]
      gyro_body  (3,) -- angular rate measurement                           [rad/s]
      mag_body   (3,) -- magnetometer (reference field = ENU +x = East)     [Gauss]
    """

    update_rate = 200.0   # [Hz]

    sigma_accel = 0.05    # [m/s²]   white noise 1-sigma per sample
    sigma_gyro  = 0.002   # [rad/s]  white noise 1-sigma per sample
    sigma_mag   = 0.01    # [Gauss]  white noise 1-sigma per sample

    _G_INERTIAL = np.array([0.0, 0.0, -9.81])
    _MAG_REF    = np.array([1.0, 0.0,  0.0])

    def _measure(self, accel_inertial, euler, wb):
        phi, theta, psi = euler
        R_T = _rot_ZYX(phi, theta, psi).T

        specific_force = accel_inertial - self._G_INERTIAL
        accel_body = R_T @ specific_force + self.sigma_accel * np.random.randn(3)
        gyro_body  = wb                   + self.sigma_gyro  * np.random.randn(3)
        mag_body   = R_T @ self._MAG_REF  + self.sigma_mag   * np.random.randn(3)

        return accel_body, gyro_body, mag_body


# ── Barometer ─────────────────────────────────────────────────────────────────

class BaroSensor(_DiscreteSensor):
    """
    Barometric altimeter.

    update_rate : 50 Hz

    Input : true_z [m]  (ENU, positive up, 0 = sea level)
    Output: baro_z [m]
    """

    update_rate = 50.0   # [Hz]
    sigma_baro  = 0.5    # [m]  white noise 1-sigma per sample

    def _measure(self, true_z):
        return true_z + self.sigma_baro * np.random.randn()


# ── GNSS ──────────────────────────────────────────────────────────────────────

class GNSSSensor(_DiscreteSensor):
    """
    GNSS receiver — position + velocity.

    update_rate : 5 Hz  (typical RTK/GNSS in challenging offshore environment)

    Inputs : true_pos (3,) [m], true_vel (3,) [m/s]
    Outputs: gnss_pos (3,) [m], gnss_vel (3,) [m/s]
    """

    update_rate = 50.0     # [Hz]
    sigma_pos   = 0.5     # [m]
    sigma_vel   = 0.1     # [m/s]

    def _measure(self, true_pos, true_vel):
        gnss_pos = true_pos + self.sigma_pos * np.random.randn(3)
        gnss_vel = true_vel + self.sigma_vel * np.random.randn(3)
        return gnss_pos, gnss_vel


# ── 2-D LiDAR ─────────────────────────────────────────────────────────────────

class LiDAR2DSensor(_DiscreteSensor):
    """
    Horizontal 2-D LiDAR — models a planar scan around the drone.

    Used to measure:
      standoff  -- radial distance from drone to turbine wall  [m]
      bearing   -- angle from drone body-x to turbine centre   [rad]

    Model:
      Beams are cast in the horizontal (XY) plane in inertial frame.
      Each beam finds the intersection with the cylindrical monopile
      (radius R_turbine, axis at origin).  Range noise is added per beam.
      Standoff and bearing are extracted from the minimum-range return,
      which corresponds to the beam pointing closest to the turbine centre.

    update_rate : 10 Hz  (e.g. RPLIDAR A3 at reduced rate for integration)

    Inputs:
      drone_pos  (3,) -- inertial [x, y, z]  [m]
      psi        float -- yaw angle            [rad]
      R_turbine  float -- turbine surface radius [m]

    Outputs:
      standoff   float -- estimated radial standoff distance   [m]
      bearing    float -- bearing to turbine centre in body frame [rad]
                          0 = directly ahead, positive = left
      ranges     (n_beams,) -- raw per-beam ranges              [m]
      beam_angles(n_beams,) -- beam angles in body frame        [rad]
    """

    update_rate = 10.0    # [Hz]
    n_beams     = 360     # angular resolution [beams per full revolution]
    max_range   = 30.0    # [m]  beams beyond this are clipped (return max_range)
    sigma_range = 0.02    # [m]  per-beam range noise (1-sigma)

    def _measure(self, drone_pos, psi, R_turbine):
        x_d, y_d = drone_pos[0], drone_pos[1]

        # Beam angles in inertial frame — full 360° scan
        beam_angles_inertial = np.linspace(0, 2 * np.pi, self.n_beams, endpoint=False)
        dx = np.cos(beam_angles_inertial)
        dy = np.sin(beam_angles_inertial)

        # Ray-cylinder intersection: ||(x_d + t*dx, y_d + t*dy)||² = R²
        # → t² + 2t(x_d·dx + y_d·dy) + (x_d² + y_d² - R²) = 0
        b   = 2.0 * (x_d * dx + y_d * dy)
        c   = x_d**2 + y_d**2 - R_turbine**2
        disc = b**2 - 4.0 * c            # a = 1

        ranges = np.full(self.n_beams, self.max_range)
        hit    = disc >= 0
        t_hit  = (-b[hit] - np.sqrt(disc[hit])) / 2.0   # smaller (nearer) root
        valid  = t_hit > 0.0
        ranges[np.where(hit)[0][valid]] = np.minimum(
            t_hit[valid], self.max_range)

        # Add range noise to hit beams
        ranges[hit] += self.sigma_range * np.random.randn(hit.sum())
        ranges = np.clip(ranges, 0.0, self.max_range)

        # Standoff from nearest return
        i_min    = np.argmin(ranges)
        standoff = ranges[i_min]                          # ≈ r_drone - R_turbine

        # Bearing to turbine centre in body frame
        # True bearing (inertial): direction from drone to origin
        bearing_inertial = np.arctan2(-y_d, -x_d)
        bearing_body     = np.arctan2(
            np.sin(bearing_inertial - psi),
            np.cos(bearing_inertial - psi))               # wrapped to ±π

        # Beam angles in body frame (for output / visualisation)
        beam_angles_body = np.arctan2(
            np.sin(beam_angles_inertial - psi),
            np.cos(beam_angles_inertial - psi))

        return standoff, bearing_body, ranges, beam_angles_body


# ── AHRS ──────────────────────────────────────────────────────────────────────

class AHRSSensor(_DiscreteSensor):
    """
    Attitude and Heading Reference System.
    Directly outputs phi, theta, psi (fused internally — black box to the EKF).

    update_rate : 100 Hz
    Input : euler (3,) -- true [phi, theta, psi] [rad]
    Output: (3,)       -- noisy euler angles      [rad]
    """

    update_rate  = 100.0                  # [Hz]
    sigma_phi    = np.radians(0.5)        # [rad]  roll  1-sigma  (~0.5°)
    sigma_theta  = np.radians(0.5)        # [rad]  pitch 1-sigma  (~0.5°)
    sigma_psi    = np.radians(1.0)        # [rad]  heading 1-sigma (~1°)

    def _measure(self, euler):
        phi, theta, psi = euler
        return np.array([
            phi   + self.sigma_phi   * np.random.randn(),
            theta + self.sigma_theta * np.random.randn(),
            psi   + self.sigma_psi   * np.random.randn(),
        ])


# ── Sensor suite (convenience wrapper) ───────────────────────────────────────

class SensorSuite:
    """
    Bundles all sensors and calls tick() on each every simulation step.

    Usage in simulation loop:
        suite = SensorSuite(R_turbine=4.0)
        ...
        meas = suite.tick(t, dt,
                          accel_inertial, euler, wb,   # IMU
                          pos, vel)                     # baro / GNSS / LiDAR

    Returns a dict with keys: 'imu', 'baro', 'gnss', 'lidar'
    Each value is the held measurement (None until first fire).
    """

    def __init__(self, R_turbine=4.0):
        self.imu   = IMUSensor()
        self.baro  = BaroSensor()
        self.gnss  = GNSSSensor()
        self.ahrs  = AHRSSensor()
        self.lidar = LiDAR2DSensor()
        self.R_turbine = R_turbine

    def tick(self, t, dt, accel_inertial, euler, wb, pos, vel):
        phi, theta, psi = euler
        return {
            'imu'  : self.imu.tick(t, dt, accel_inertial, euler, wb),
            'baro' : self.baro.tick(t, dt, pos[2]),
            'gnss' : self.gnss.tick(t, dt, pos, vel),
            'ahrs' : self.ahrs.tick(t, dt, euler),
            'lidar': self.lidar.tick(t, dt, pos, psi, self.R_turbine),
        }


# ── noise covariance matrices (for Kalman filter R matrix) ───────────────────

def imu_R():
    """Measurement noise covariance for IMU accel + gyro (6×6)."""
    s = np.concatenate([
        np.full(3, IMUSensor.sigma_accel**2),
        np.full(3, IMUSensor.sigma_gyro**2),
    ])
    return np.diag(s)

def baro_R():
    """Measurement noise variance for barometer (scalar)."""
    return BaroSensor.sigma_baro**2

def gnss_R():
    """Measurement noise covariance for GNSS pos + vel (6×6)."""
    s = np.concatenate([
        np.full(3, GNSSSensor.sigma_pos**2),
        np.full(3, GNSSSensor.sigma_vel**2),
    ])
    return np.diag(s)

def lidar_R():
    """Measurement noise covariance for LiDAR standoff + bearing (2×2)."""
    return np.diag([LiDAR2DSensor.sigma_range**2, (0.5 * np.pi/180)**2])
