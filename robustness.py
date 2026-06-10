"""
robustness.py  —  Controller robustness analysis for SC and UW sims
====================================================================

Usage
-----
  python robustness.py --vehicle sc  --test step_z       --analysis single
  python robustness.py --vehicle sc  --test step_z       --analysis sweep
  python robustness.py --vehicle sc  --test disturbance  --analysis mc
  python robustness.py --vehicle uw  --test step_z       --analysis single --no-ekf
  python robustness.py --vehicle uw  --test turn         --analysis sweep
  python robustness.py --vehicle both --test step_z      --analysis single

Tests (--test)
--------------
  step_z       Vertical step input.  Metrics: rise time, settling time, overshoot.
  step_xy      Horizontal (radial) step input.  Same metrics.
  turn         Azimuth/yaw step.  Same metrics, tracked on yaw angle.
  disturbance  Constant lateral force applied while holding position.
               Metrics: steady-state position error, recovery time.

Analysis (--analysis)
---------------------
  single   One run.  Prints metrics and plots the time response.
  sweep    Tornado chart: vary one parameter at a time, track primary metric.
  mc       Monte Carlo: randomise all parameters, histogram of primary metric.

EKF (optional)
--------------
  --ekf      Force EKF on  (override file default)
  --no-ekf   Force EKF off (override file default)
  (omit)     Use whatever the sim file has set

Sweep / MC configuration
------------------------
  Edit SWEEP_PARAMS, MC_RUNS, MC_*_STD constants below.
"""

import argparse
import importlib
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

FAILS_DIR = "Fails"

# ─────────────────────────────────────────────────────────────────────────────
#  RUN CONFIGURATION  — edit these and just run the file (no terminal flags needed)
# ─────────────────────────────────────────────────────────────────────────────

# Vehicle:   'sc'   | 'uw'   | 'both'
VEHICLE  = 'sc'

# Test:      'step_z' | 'step_xy' | 'turn' | 'disturbance' | 'gust'
TEST     = 'step_z'

# Analysis:  'single' | 'sweep' | 'mc'
ANALYSIS = 'sweep'

# EKF:       True (force on) | False (force off) | None (use sim default)
EKF      = None

# Step plot window: seconds shown AFTER the step (step always appears at t=1)
STEP_DURATION = 10.0

# Rise time thresholds [%] — time for signal to go from RISE_LO to RISE_HI of the step
RISE_LO_PCT = 10.0
RISE_HI_PCT = 90.0

# Settling band [%] — shaded region around x_final in the step plot
SETTLING_BAND_PCT = 5.0

# Show a time-response plot after each individual sim in sweep/MC (blocks until closed)
SWEEP_SHOW_PLOTS = False

# ─────────────────────────────────────────────────────────────────────────────
#  Sweep and Monte Carlo settings
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_PARAMS = [
    ("Mass",       "mass_factor"),
    ("Ixx",        "ixx_factor"),
    ("Iyy",        "iyy_factor"),
    ("Izz",        "izz_factor"),
    ("Motor lag",  "tau_m_factor"),
]

# Adaptive sweep: step up from nominal until fail, step down to SWEEP_MIN
SWEEP_STEP           = 0.10   # factor increment per step (e.g. 0.10 → 10 % steps)
SWEEP_MIN            = 0.30   # lower bound for downward sweep (30 % of nominal)
SWEEP_FAIL_WINDOW_S  = 5.0    # seconds at end of sim used to judge settling
# Fail = mean absolute error of signal vs x_final in that window > settling band

MC_RUNS        = 100
MC_SEED        = 42
MC_MASS_STD    = 0.10   # ±10 % std
MC_INERTIA_STD = 0.15   # ±15 % std per axis
MC_TAU_STD     = 0.20   # ±20 % std

# ─────────────────────────────────────────────────────────────────────────────
#  Test configuration
#  Each entry defines the sim overrides + how to extract the tracked signal.
#
#  'extract'  : fn(X) → 1-D array of tracked variable over time
#  'x_final'  : commanded final value (for rise/settling metrics)
#  'step_t'   : time at which step is applied [s] (used to find x_init)
#  'dist_F'   : (disturbance test only) force magnitude [N]
# ─────────────────────────────────────────────────────────────────────────────

TEST_CONFIGS = {
    # ── SC ──────────────────────────────────────────────────────────────────
    # All SC step/turn tests use cylindrical ndarray [t, r, θ, z, ṙ, θ̇, ż].
    # This sets _USE_CYL_REF=True so the event-triggered waypoint manager,
    # cylindrical PID, and EKF all operate identically to normal flight.
    # Velocity profiles are trapezoidal (ramp-up / cruise / ramp-down).
    ('step_z', 'sc'): {
        'label':    'Vertical step  (SC)',
        'overrides': {
            'traj_mode':     'custom',
            # 10 m climb: z=0→10, v_max=2 m/s, 1 s ramp each side
            # ramp dz=1 m, cruise dz=8 m (4 s), ramp dz=1 m  → total 10 m
            'traj_segments': np.array([
                [0.0,  6., 0.,  0.,  0., 0., 0.],   # hover
                [3.0,  6., 0.,  0.,  0., 0., 0.],   # hold before step
                [4.0,  6., 0.,  1.,  0., 0., 2.],   # ramp-up done
                [8.0,  6., 0.,  9.,  0., 0., 2.],   # cruise done
                [9.0,  6., 0., 10.,  0., 0., 0.],   # decel done
                [45.0, 6., 0., 10.,  0., 0., 0.],   # hold at target
            ]),
            't_buffer':      45.0,
            'dist_enabled':  False,
            'wind_enabled':  False,
        },
        'extract':  lambda X: X[2],        # z position
        'x_final':  10.0,
        'step_t':   3.0,
        'ylabel':   'z  [m]',
        'metric':   'rise_time',
    },
    ('step_xy', 'sc'): {
        'label':    'Horizontal radial step  (SC)',
        'overrides': {
            'traj_mode':     'custom',
            # 3 m radial step: r=6→9, v_max=1.5 m/s, 1 s ramp each side
            # ramp dr=0.75 m, cruise dr=1.5 m (1 s), ramp dr=0.75 m
            'traj_segments': np.array([
                [0.0,  6.,    0., 5., 0.,   0., 0.],  # hover
                [3.0,  6.,    0., 5., 0.,   0., 0.],  # hold
                [4.0,  6.75,  0., 5., 1.5,  0., 0.],  # ramp-up done
                [5.0,  8.25,  0., 5., 1.5,  0., 0.],  # cruise done
                [6.0,  9.,    0., 5., 0.,   0., 0.],  # decel done
                [45.0, 9.,    0., 5., 0.,   0., 0.],  # hold
            ]),
            't_buffer':      45.0,
            'dist_enabled':  False,
            'wind_enabled':  False,
        },
        'extract':  lambda X: X[0],        # x position (= r at θ=0)
        'x_final':  9.0,
        'step_t':   3.0,
        'ylabel':   'x  [m]',
        'metric':   'rise_time',
    },
    ('turn', 'sc'): {
        'label':    'Yaw / azimuth step  (SC)',
        'overrides': {
            'traj_mode':     'custom',
            # 30° azimuth turn at r=6, z=5: triangular θ profile (no cruise)
            # vθ_max=0.524 rad/s (30 deg/s), t_ramp=0.5 s each side
            # ramp dθ=0.131 rad, cruise dθ=0.262 rad (0.5 s), ramp dθ=0.131 rad
            'traj_segments': np.array([
                [0.0,  6., 0.,     5., 0., 0.,     0.],  # hover
                [3.0,  6., 0.,     5., 0., 0.,     0.],  # hold
                [3.5,  6., 0.131,  5., 0., 0.524,  0.],  # ramp-up done
                [4.0,  6., 0.393,  5., 0., 0.524,  0.],  # cruise done
                [4.5,  6., 0.524,  5., 0., 0.,     0.],  # decel done (θ=30°)
                [45.0, 6., 0.524,  5., 0., 0.,     0.],  # hold
            ]),
            't_buffer':      45.0,
            'dist_enabled':  False,
            'wind_enabled':  False,
        },
        'extract':  lambda X: np.degrees(np.arctan2(X[1], X[0])),  # azimuth [deg]
        'x_final':  30.0,
        'step_t':   3.0,
        'ylabel':   'azimuth  [deg]',
        'metric':   'max_yaw_rate_dps',
    },
    ('disturbance', 'sc'): {
        'label':    'Constant lateral wind  (SC)',
        'overrides': {
            'traj_mode':     'hold',
            'wind_speed':    15.0,   # m/s constant — REQ-SNC-01
            'traj_segments': None,
        },
        'extract':  lambda X: np.sqrt(X[0]**2 + X[1]**2),   # horizontal offset
        'x_final':  0.0,
        'step_t':   5.0,
        'ylabel':   'horizontal offset  [m]',
        'metric':   'ss_error',
    },
    ('gust', 'sc'): {
        'label':    'Gust response  (SC)',
        'overrides': {
            'traj_mode':     'hold',
            'gust_force':    15.0,   # 0.5 s pulse at t=5 — REQ-SNC-02
            'traj_segments': None,
        },
        'extract':  lambda X: np.degrees(X[3]),   # roll [deg]
        'x_final':  0.0,
        'step_t':   5.0,
        'ylabel':   'roll  [deg]',
        'metric':   'max_att_deg',
    },

    # ── UW ──────────────────────────────────────────────────────────────────
    ('step_z', 'uw'): {
        'label':    'Depth step  (UW)',
        'overrides': {
            'traj_mode':     'custom',
            # cylindrical (t, r, theta_rad, z); z negative = down
            'traj_segments': [(0, 6, 0, -5), (3, 6, 0, -15)],
            't_buffer':      60.0,
            'dist_enabled':  False,
            'current_enabled': False,
        },
        'extract':  lambda X: X[2],        # z (depth, negative)
        'x_final':  -15.0,
        'step_t':   3.0,
        'ylabel':   'z  [m]  (−ve = down)',
        'metric':   'rise_time',
    },
    ('step_xy', 'uw'): {
        'label':    'Horizontal radial step  (UW)',
        'overrides': {
            'traj_mode':     'custom',
            'traj_segments': [(0, 6, 0, -10), (3, 9, 0, -10)],
            't_buffer':      60.0,
            'dist_enabled':  False,
            'current_enabled': False,
        },
        'extract':  lambda X: np.sqrt(X[0]**2 + X[1]**2),  # radial distance
        'x_final':  9.0,
        'step_t':   3.0,
        'ylabel':   'radial distance  [m]',
        'metric':   'rise_time',
    },
    ('turn', 'uw'): {
        'label':    'Heading step  (UW)',
        'overrides': {
            'traj_mode':     'custom',
            'traj_segments': [(0, 6, 0, -10), (3, 6, np.pi/6, -10)],
            't_buffer':      60.0,
            'dist_enabled':  False,
            'current_enabled': False,
        },
        'extract':  lambda X: np.degrees(X[5]),   # yaw [deg]
        'x_final':  30.0,
        'step_t':   3.0,
        'ylabel':   'yaw  [deg]',
        'metric':   'heading_error_deg',
    },
    ('gust', 'uw'): {
        'label':    'Gust response  (UW)',
        'overrides': {
            'traj_mode':     'hold',
            'gust_force':    15.0,   # 0.5 s pulse at t=5
            'traj_segments': None,
        },
        'extract':  lambda X: np.degrees(X[3]),   # roll [deg]
        'x_final':  0.0,
        'step_t':   5.0,
        'ylabel':   'roll  [deg]',
        'metric':   'max_att_deg',
    },
    ('disturbance', 'uw'): {
        'label':    'Constant lateral current  (UW)',
        'overrides': {
            'traj_mode':     'hold',
            'current_enabled': True,
            'current_speed':   1.5,       # m/s — REQ-SNC-06 value
            't_buffer':      120.0,
        },
        'extract':  lambda X: np.sqrt(X[0]**2 + X[1]**2),
        'x_final':  0.0,
        'step_t':   5.0,
        'ylabel':   'horizontal offset  [m]',
        'metric':   'ss_error',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  Metric extraction utilities
# ─────────────────────────────────────────────────────────────────────────────

def _rise_time(t, x, x_init, x_final, pct_lo=0.10, pct_hi=0.90):
    """Time for signal to travel from pct_lo to pct_hi of the step."""
    step = x_final - x_init
    if abs(step) < 1e-6:
        return np.nan
    lo = x_init + pct_lo * step
    hi = x_init + pct_hi * step
    if step > 0:
        i_lo = np.argmax(x >= lo)
        i_hi = np.argmax(x >= hi)
    else:
        i_lo = np.argmax(x <= lo)
        i_hi = np.argmax(x <= hi)
    if i_lo == 0 or i_hi == 0 or i_hi <= i_lo:
        return np.nan
    return float(t[i_hi] - t[i_lo])


def _settling_time(t, x, x_final, band=0.02):
    """Last time signal is outside ±band fraction of step amplitude from x_final."""
    amp = abs(x[-1] - x[0])
    if amp < 1e-6:
        return np.nan
    tol = band * amp
    outside = np.where(np.abs(x - x_final) > tol)[0]
    if len(outside) == 0:
        return float(t[0])
    return float(t[outside[-1]])


def _overshoot(x, x_init, x_final):
    """% overshoot relative to step size (0 if no overshoot)."""
    step = x_final - x_init
    if abs(step) < 1e-6:
        return 0.0
    if step > 0:
        return float(max((np.max(x) - x_final) / step * 100, 0.0))
    else:
        return float(max((x_final - np.min(x)) / abs(step) * 100, 0.0))


def _ss_error(t, x, x_final, last_frac=0.15):
    """Mean absolute deviation from x_final over the last fraction of time."""
    n = max(int(len(t) * last_frac), 1)
    return float(np.mean(np.abs(x[-n:] - x_final)))


def _compute_metrics(t, X, cfg):
    """Return dict of metrics for one sim run."""
    sig     = cfg['extract'](X)
    step_t  = cfg['step_t']
    x_final = cfg['x_final']
    # x_init = mean of signal before the step
    pre_mask = t < step_t
    x_init   = float(np.mean(sig[pre_mask])) if pre_mask.any() else float(sig[0])

    rt  = _rise_time(t, sig, x_init, x_final, RISE_LO_PCT / 100.0, RISE_HI_PCT / 100.0)
    st  = _settling_time(t, sig, x_final)
    os  = _overshoot(sig, x_init, x_final)
    sse = _ss_error(t, sig, x_final)

    # Attitude metrics — available when X has ≥12 rows (SC/UW both do)
    if X.shape[0] >= 12:
        roll_deg  = np.degrees(X[3])
        pitch_deg = np.degrees(X[4])
        yaw_rate  = np.degrees(X[11])  # row 11 = yaw rate [rad/s]
        max_roll   = float(np.max(np.abs(roll_deg)))
        max_pitch  = float(np.max(np.abs(pitch_deg)))
        max_att    = float(max(max_roll, max_pitch))
        max_yr     = float(np.max(np.abs(yaw_rate)))
    else:
        max_roll = max_pitch = max_att = max_yr = np.nan

    # heading_error_deg: mean absolute deviation of yaw from x_final (for turn test)
    post_mask = t > (step_t + 5.0)
    if post_mask.any() and X.shape[0] >= 6:
        yaw_sig = np.degrees(X[5]) if X.shape[0] > 5 else sig
        heading_err = float(np.mean(np.abs(yaw_sig[post_mask] - x_final)))
    else:
        heading_err = np.nan

    # Fail = mean absolute error of signal vs x_final over last SWEEP_FAIL_WINDOW_S seconds
    amp = abs(x_final - x_init)
    if amp > 1e-6:
        band      = (SETTLING_BAND_PCT / 100.0) * amp
        t_end     = t[-1]
        win_mask  = t >= (t_end - SWEEP_FAIL_WINDOW_S)
        window    = sig[win_mask] if win_mask.any() else sig[-1:]
        mean_err  = float(np.mean(np.abs(window - x_final)))
        failed    = mean_err > band
    else:
        failed = False

    return {
        'rise_time':         rt,
        'settling_time':     st,
        'overshoot_pct':     os,
        'ss_error':          sse,
        'max_roll_deg':      max_roll,
        'max_pitch_deg':     max_pitch,
        'max_att_deg':       max_att,
        'max_yaw_rate_dps':  max_yr,
        'heading_error_deg': heading_err,
        'x_init':            x_init,
        'signal':            sig,
        'failed':            failed,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Loader + safe runner
# ─────────────────────────────────────────────────────────────────────────────

def _load(vehicle):
    if vehicle == 'sc':
        import quadcopterSC
        importlib.reload(quadcopterSC)
        return quadcopterSC.run_sim, 'SC'
    elif vehicle == 'uw':
        import quadcopterUW
        importlib.reload(quadcopterUW)
        return quadcopterUW.run_sim, 'UW'
    raise ValueError(f"Unknown vehicle '{vehicle}'")


def _run(run_fn, overrides, timeseries=False):
    """Safe wrapper — always closes figures and returns a result dict."""
    ov = dict(overrides)
    if timeseries:
        ov['return_timeseries'] = True
    try:
        result = run_fn(overrides=ov, show_plots=False)
    except Exception as exc:
        print(f"    [WARN] sim raised {type(exc).__name__}: {exc}")
        result = {'rms_pos_error': np.nan, 'max_pos_error': np.nan,
                  'crashed': True, 't_complete': None,
                  'max_att_deg': np.nan, 'max_yaw_rate_dps': np.nan}
    plt.close('all')   # prevent figure accumulation across runs
    return result


def _build_overrides(cfg, ekf_flag, extra=None):
    """Merge test overrides + EKF flag + any extras."""
    ov = {k: v for k, v in cfg['overrides'].items() if v is not None}
    if ekf_flag is not None:
        ov['ekf_enabled'] = ekf_flag
    if extra:
        ov.update(extra)
    return ov


def _clip_normal(rng, mean, std, n):
    return np.clip(rng.normal(mean, std, size=n), mean - 3*std, mean + 3*std)


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis modes
# ─────────────────────────────────────────────────────────────────────────────

def _plot_response(cfg, metrics, result, title, save_path=None, block=True):
    """Plot the step time response. Blocks until window closed if block=True."""
    t, sig  = result['t'], metrics['signal']
    step_t  = cfg['step_t']
    x_init  = metrics['x_init']
    x_final = cfg['x_final']
    amp     = abs(x_final - x_init)

    _t_end = 1.0 + STEP_DURATION
    t_plot = t - step_t + 1.0
    mask   = (t_plot >= 0.0) & (t_plot <= _t_end)
    t_plt  = t_plot[mask]
    s_plt  = sig[mask]

    fig, ax = plt.subplots(figsize=(10, 4))
    _t_ref = np.array([0.0, 1.0, 1.0, _t_end])
    _v_ref = np.array([x_init, x_init, x_final, x_final])
    ax.plot(_t_ref, _v_ref, color='red', lw=1.5, ls='--', label='Reference')
    ax.plot(t_plt, s_plt, color='steelblue', lw=1.8, label='True')

    if amp > 1e-6:
        _band = SETTLING_BAND_PCT / 100.0
        ax.axhspan(x_final - _band*amp, x_final + _band*amp,
                   color='green', alpha=0.15,
                   label=f'Settling band (±{SETTLING_BAND_PCT:.4g}%)')

    if not np.isnan(metrics['rise_time']) and amp > 1e-6:
        _hi   = x_init + (RISE_HI_PCT / 100.0) * (x_final - x_init)
        _post = t_plt >= 1.0
        _sp   = s_plt[_post]
        _tp   = t_plt[_post]
        _i_hi = np.argmax(_sp >= _hi) if (x_final > x_init) else np.argmax(_sp <= _hi)
        _t_rise = _tp[_i_hi]
        ax.axvline(_t_rise, color='darkorange', lw=1.4, ls=':',
                   label=f'Rise time ({metrics["rise_time"]:.2f} s)')

    if result['crashed']:
        ax.set_facecolor('#fff0f0')
        ax.set_title(title + '  [CRASH]', color='red')
    else:
        ax.set_title(title)

    ax.set_xlim(0.0, _t_end)
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel(cfg['ylabel'])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if block:
        plt.show(block=False)
        plt.pause(0.1)
        while plt.fignum_exists(fig.number):
            plt.pause(0.2)
    else:
        plt.show(block=False)
        plt.pause(0.1)
    return fig


def single(vehicle, test, ekf_flag):
    """One run: print metrics and plot time response."""
    key = (test, vehicle)
    if key not in TEST_CONFIGS:
        print(f"No test config for ({test}, {vehicle})")
        return

    cfg    = TEST_CONFIGS[key]
    run_fn, lbl = _load(vehicle)
    ov     = _build_overrides(cfg, ekf_flag)

    print(f"\n{'='*60}")
    print(f"  SINGLE RUN — {cfg['label']}  [{lbl}]")
    ekf_str = {True: 'EKF ON', False: 'EKF OFF', None: 'EKF default'}[ekf_flag]
    print(f"  {ekf_str}")
    print(f"{'='*60}")

    result  = _run(run_fn, ov, timeseries=True)

    if 't' not in result:
        print("  ERROR: sim did not return timeseries. Check return_timeseries support.")
        return

    t, X    = result['t'], result['X']
    metrics = _compute_metrics(t, X, cfg)

    def _fmt(val, unit='', fmt='.3f'):
        return f"{val:{fmt}} {unit}".strip() if not np.isnan(val) else "N/A"

    print(f"  Rise time (10→90%):   {_fmt(metrics['rise_time'], 's')}")
    print(f"  Settling time (2%):   {_fmt(metrics['settling_time'], 's')}")
    print(f"  Overshoot:            {_fmt(metrics['overshoot_pct'], '%', '.1f')}")
    print(f"  Steady-state error:   {_fmt(metrics['ss_error'], '', '.4f')}")
    print(f"  Max roll:             {_fmt(metrics['max_roll_deg'], 'deg', '.2f')}")
    print(f"  Max pitch:            {_fmt(metrics['max_pitch_deg'], 'deg', '.2f')}")
    print(f"  Max attitude:         {_fmt(metrics['max_att_deg'], 'deg', '.2f')}")
    print(f"  Max yaw rate:         {_fmt(metrics['max_yaw_rate_dps'], 'deg/s', '.2f')}")
    print(f"  Heading error:        {_fmt(metrics['heading_error_deg'], 'deg', '.2f')}")
    print(f"  Crashed:              {result['crashed']}")

    out = f"response_{vehicle}_{test}.png"
    _plot_response(cfg, metrics, result,
                   title=f"{cfg['label']}  —  {ekf_str}",
                   save_path=out, block=True)
    print(f"\n  Plot saved → {out}")


def sweep(vehicle, test, ekf_flag):
    """Adaptive sweep: step each parameter up until fail, down to SWEEP_MIN."""
    key = (test, vehicle)
    if key not in TEST_CONFIGS:
        print(f"No test config for ({test}, {vehicle})")
        return

    cfg     = TEST_CONFIGS[key]
    run_fn, lbl = _load(vehicle)
    primary = cfg['metric']

    if os.path.exists(FAILS_DIR):
        shutil.rmtree(FAILS_DIR)
    os.makedirs(FAILS_DIR)

    print(f"\n{'='*60}")
    print(f"  ADAPTIVE SWEEP — {cfg['label']}  [{lbl}]")
    print(f"  step={SWEEP_STEP:.2g}  min={SWEEP_MIN:.2g}  fail_window={SWEEP_FAIL_WINDOW_S}s")
    print(f"{'='*60}")

    def _one_run(v, key_p, sim_label):
        print(f"  {sim_label}  {key_p}={v:.3f} ...", end='\r', flush=True)
        ov  = _build_overrides(cfg, ekf_flag, {key_p: v})
        res = _run(run_fn, ov, timeseries=True)
        if 't' in res:
            m      = _compute_metrics(res['t'], res['X'], cfg)
            failed = m['failed']
            val    = m[primary] if not failed else np.nan
            tag    = "  FAIL" if failed else ""
            print(f"  {sim_label}  {key_p}={v:.3f}  {primary}={val:.4f}{tag}")
            save_path = os.path.join(FAILS_DIR, f"fail_{vehicle}_{test}_{key_p}_{v:.3g}.png") if failed else None
            if failed or SWEEP_SHOW_PLOTS:
                _plot_response(cfg, m, res,
                               title=f"{cfg['label']}  {sim_label}  {key_p}={v:.3f}{tag}",
                               save_path=save_path,
                               block=SWEEP_SHOW_PLOTS)
        else:
            m      = None
            failed = True
            val    = np.nan
            print(f"  {sim_label}  {key_p}={v:.3f}  ERROR")
        return val, failed, m

    param_data   = []
    sensitivities = []
    sim_i        = 0

    for param_label, key_p in SWEEP_PARAMS:
        print(f"\n── {param_label} ──")
        all_vals, all_rt, all_st, all_failed = [], [], [], []

        # ── nominal ──
        sim_i += 1
        val, failed, m = _one_run(1.0, key_p, f"[#{sim_i}  nom]")
        all_vals.append(1.0)
        all_rt.append(m['rise_time']     if m and not failed else np.nan)
        all_st.append(m['settling_time'] if m and not failed else np.nan)
        all_failed.append(failed)
        nom_val = val

        # ── sweep UP until fail ──
        v = round(1.0 + SWEEP_STEP, 10)
        while True:
            sim_i += 1
            val, failed, m = _one_run(v, key_p, f"[#{sim_i}   up]")
            all_vals.append(v)
            all_rt.append(m['rise_time']     if m and not failed else np.nan)
            all_st.append(m['settling_time'] if m and not failed else np.nan)
            all_failed.append(failed)
            if failed:
                break
            v = round(v + SWEEP_STEP, 10)

        # ── sweep DOWN to SWEEP_MIN ──
        v = round(1.0 - SWEEP_STEP, 10)
        while v >= SWEEP_MIN - 1e-9:
            sim_i += 1
            val, failed, m = _one_run(v, key_p, f"[#{sim_i} down]")
            all_vals.append(v)
            all_rt.append(m['rise_time']     if m and not failed else np.nan)
            all_st.append(m['settling_time'] if m and not failed else np.nan)
            all_failed.append(failed)
            if failed:
                break
            v = round(v - SWEEP_STEP, 10)

        # sort by parameter value for clean plots
        order = np.argsort(all_vals)
        all_vals   = [all_vals[i]   for i in order]
        all_rt     = [all_rt[i]     for i in order]
        all_st     = [all_st[i]     for i in order]
        all_failed = [all_failed[i] for i in order]

        primary_vals = [rt if not f else np.nan for rt, f in zip(all_rt, all_failed)]
        nom_primary  = primary_vals[all_vals.index(1.0)] if 1.0 in all_vals else np.nan
        finite = [v for v in primary_vals if not np.isnan(v)]
        d_min  = min(finite) - nom_primary if finite else 0.0
        d_max  = max(finite) - nom_primary if finite else 0.0
        sensitivities.append((param_label, d_min, d_max))
        param_data.append((param_label, all_vals, all_rt, all_st, all_failed))

    # ── Tornado chart ──
    sensitivities.sort(key=lambda s: abs(s[2] - s[1]), reverse=True)
    fig, ax = plt.subplots(figsize=(10, max(3, len(sensitivities) * 0.9 + 1)))
    for i, (_, d_min, d_max) in enumerate(sensitivities):
        if d_max > 0:
            ax.barh(i, d_max, color='steelblue', alpha=0.8, height=0.5)
        if d_min < 0:
            ax.barh(i, abs(d_min), left=d_min, color='tomato', alpha=0.8, height=0.5)
    ax.set_yticks(range(len(sensitivities)))
    ax.set_yticklabels([s[0] for s in sensitivities])
    ax.axvline(0, color='black', lw=1.2)
    ax.set_xlabel(f"Δ{primary}  (relative to nominal)")
    ax.set_title(f"Tornado: {cfg['label']}  [{lbl}]")
    fig.tight_layout()
    plt.show()

    # ── Per-parameter metric plots (rise time & settling time) ──
    n = len(param_data)
    fig2, axes2 = plt.subplots(n, 2, figsize=(12, 2.8 * n), squeeze=False)
    fig2.suptitle(f"Sweep metrics — {cfg['label']}  [{lbl}]", fontsize=12)

    for row, (plabel, pvals, rts, sts, fails) in enumerate(param_data):
        x     = np.arange(len(pvals))
        xlbls = [f"{v:.3g}" for v in pvals]
        colors = ['tomato' if f else 'steelblue' for f in fails]

        for col, (metric_vals, metric_name) in enumerate([
            (rts, f'Rise time  [{RISE_LO_PCT:.4g}→{RISE_HI_PCT:.4g}%]  [s]'),
            (sts, f'Settling time  [±{SETTLING_BAND_PCT:.4g}%]  [s]'),
        ]):
            ax = axes2[row, col]
            bars = ax.bar(x, [v if not np.isnan(v) else 0 for v in metric_vals],
                          color=colors, alpha=0.85, edgecolor='white')
            for bi, (bar, failed_flag) in enumerate(zip(bars, fails)):
                if failed_flag:
                    bar.set_hatch('//')
                    ax.text(bi, 0.02, 'FAIL', ha='center', va='bottom',
                            fontsize=7, color='darkred', rotation=90)
            ax.set_xticks(x)
            ax.set_xticklabels(xlbls, fontsize=8)
            ax.set_xlabel(f"{plabel}  (factor)", fontsize=9)
            ax.set_ylabel(metric_name, fontsize=8)
            ax.grid(True, axis='y', alpha=0.35)
            if col == 0:
                ax.set_title(plabel, fontsize=9, fontweight='bold')

    fig2.tight_layout()
    plt.show()


def mc(vehicle, test, ekf_flag):
    """Monte Carlo: randomise parameters, histogram of primary metric."""
    key = (test, vehicle)
    if key not in TEST_CONFIGS:
        print(f"No test config for ({test}, {vehicle})")
        return

    cfg    = TEST_CONFIGS[key]
    run_fn, lbl = _load(vehicle)
    primary = cfg['metric']

    print(f"\n{'='*60}")
    print(f"  MONTE CARLO ({MC_RUNS} runs) — {cfg['label']}  [{lbl}]")
    print(f"  Primary metric: {primary}")
    print(f"{'='*60}")

    rng    = np.random.default_rng(MC_SEED)
    mass_f = _clip_normal(rng, 1.0, MC_MASS_STD,    MC_RUNS)
    ixx_f  = _clip_normal(rng, 1.0, MC_INERTIA_STD, MC_RUNS)
    iyy_f  = _clip_normal(rng, 1.0, MC_INERTIA_STD, MC_RUNS)
    izz_f  = _clip_normal(rng, 1.0, MC_INERTIA_STD, MC_RUNS)
    tau_f  = _clip_normal(rng, 1.0, MC_TAU_STD,     MC_RUNS)

    metric_list, crash_list = [], []

    for i in range(MC_RUNS):
        print(f"  [sim {i+1}/{MC_RUNS}] running ...", end='\r', flush=True)
        ov = _build_overrides(cfg, ekf_flag, {
            'mass_factor':  mass_f[i], 'ixx_factor': ixx_f[i],
            'iyy_factor':   iyy_f[i],  'izz_factor': izz_f[i],
            'tau_m_factor': tau_f[i],
        })
        res = _run(run_fn, ov, timeseries=True)
        if 't' in res and not res['crashed']:
            m   = _compute_metrics(res['t'], res['X'], cfg)
            val = m[primary]
        else:
            val = np.nan
        print(f"  [sim {i+1:3d}/{MC_RUNS}]  m×{mass_f[i]:.2f}  "
              f"{primary}={val:.4f}  {'CRASH' if res['crashed'] else 'ok'}")
        metric_list.append(val)
        crash_list.append(bool(res['crashed']))

    arr        = np.array(metric_list)
    crash_rate = np.mean(crash_list) * 100.0
    valid      = ~np.isnan(arr)

    print(f"\n  {np.sum(valid)}/{MC_RUNS} valid runs,  crash rate {crash_rate:.1f} %")
    if valid.any():
        print(f"  {primary}: mean={np.nanmean(arr):.4f}  "
              f"std={np.nanstd(arr):.4f}  p95={np.nanpercentile(arr, 95):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    if valid.any():
        ax.hist(arr[valid], bins=20, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(np.nanmean(arr),           color='red',        lw=1.5, ls='--', label='mean')
        ax.axvline(np.nanpercentile(arr, 95), color='darkorange', lw=1.5, ls=':',  label='p95')
        ax.legend(fontsize=9)
    ax.set_xlabel(primary)
    ax.set_ylabel("Count")
    ax.set_title(f"MC {primary} — {cfg['label']}")

    ax2 = axes[1]
    ok_n, cr_n = MC_RUNS - int(sum(crash_list)), int(sum(crash_list))
    bars = ax2.bar(["Completed", "Crashed"], [ok_n, cr_n],
                   color=['steelblue', 'tomato'], edgecolor='white', alpha=0.85)
    for bar, val in zip(bars, [ok_n, cr_n]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(val), ha='center', va='bottom', fontsize=10)
    ax2.set_title(f"Pass / Crash  ({crash_rate:.1f} %)")

    fig.suptitle(f"Monte Carlo: {cfg['label']}  [{lbl}]  ({MC_RUNS} runs)", fontsize=12)
    fig.tight_layout()
    out = f"mc_{vehicle}_{test}.png"
    fig.savefig(out, dpi=150)
    plt.show()
    print(f"  MC plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys

    # If called with CLI args, parse them; otherwise fall back to the config block above.
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Quadcopter controller robustness analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__,
        )
        parser.add_argument('--vehicle',  choices=['sc', 'uw', 'both'], default='sc')
        parser.add_argument('--test',     choices=['step_z', 'step_xy', 'turn', 'disturbance', 'gust'],
                            required=True)
        parser.add_argument('--analysis', choices=['single', 'sweep', 'mc'], default='single')
        ekf_grp = parser.add_mutually_exclusive_group()
        ekf_grp.add_argument('--ekf',    dest='ekf', action='store_true',  default=None)
        ekf_grp.add_argument('--no-ekf', dest='ekf', action='store_false')
        args     = parser.parse_args()
        vehicle  = args.vehicle
        test     = args.test
        analysis = args.analysis
        ekf_flag = args.ekf
    else:
        vehicle  = VEHICLE
        test     = TEST
        analysis = ANALYSIS
        ekf_flag = EKF

    vehicles = ['sc', 'uw'] if vehicle == 'both' else [vehicle]

    for veh in vehicles:
        if analysis == 'single':
            single(veh, test, ekf_flag)
        elif analysis == 'sweep':
            sweep(veh, test, ekf_flag)
        elif analysis == 'mc':
            mc(veh, test, ekf_flag)


if __name__ == '__main__':
    main()
