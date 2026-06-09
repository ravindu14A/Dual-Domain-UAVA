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
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
#  Sweep and Monte Carlo settings
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_PARAMS = [
    ("Mass  ±25 %",      "mass_factor",  [0.75, 0.875, 1.0, 1.125, 1.25]),
    ("Ixx   ±40 %",      "ixx_factor",   [0.60, 0.80,  1.0, 1.20,  1.40]),
    ("Iyy   ±40 %",      "iyy_factor",   [0.60, 0.80,  1.0, 1.20,  1.40]),
    ("Izz   ±40 %",      "izz_factor",   [0.60, 0.80,  1.0, 1.20,  1.40]),
    ("Motor lag  ±50 %", "tau_m_factor", [0.50, 0.75,  1.0, 1.25,  1.50]),
]

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
    ('step_z', 'sc'): {
        'label':    'Vertical step  (SC)',
        'overrides': {
            'traj_mode':     'custom',
            'traj_segments': [(0, 0, 0, 0), (3, 0, 0, 10)],
            't_buffer':      45.0,
            'dist_enabled':  False,
            'wind_enabled':  False,
        },
        'extract':  lambda X: X[2],        # z position
        'x_final':  10.0,
        'step_t':   3.0,
        'ylabel':   'z  [m]',
        'metric':   'rise_time',           # primary metric for sweep/MC
    },
    ('step_xy', 'sc'): {
        'label':    'Horizontal radial step  (SC)',
        'overrides': {
            'traj_mode':     'custom',
            'traj_segments': [(0, 6, 0, 5), (3, 9, 0, 5)],
            't_buffer':      45.0,
            'dist_enabled':  False,
            'wind_enabled':  False,
        },
        'extract':  lambda X: X[0],        # x position (at theta=0, r=x)
        'x_final':  9.0,
        'step_t':   3.0,
        'ylabel':   'x  [m]',
        'metric':   'rise_time',
    },
    ('turn', 'sc'): {
        'label':    'Yaw / azimuth step  (SC)',
        'overrides': {
            'traj_mode':     'custom',
            # x,y step equivalent to 30° azimuth turn at r=6
            'traj_segments': [(0, 6, 0, 5), (3, 5.196, 3.0, 5)],
            't_buffer':      45.0,
            'dist_enabled':  False,
            'wind_enabled':  False,
        },
        'extract':  lambda X: np.degrees(np.arctan2(X[1], X[0])),  # azimuth [deg]
        'x_final':  30.0,
        'step_t':   3.0,
        'ylabel':   'azimuth  [deg]',
        'metric':   'rise_time',
    },
    ('disturbance', 'sc'): {
        'label':    'Constant lateral disturbance  (SC)',
        'overrides': {
            'traj_mode':     'hold',
            'dist_enabled':  True,
            # Constant 15 N lateral force starting at t=5 s
            'traj_segments': None,   # hold mode ignores segments
        },
        'dist_F':   15.0,            # [N] — sets DISTURBANCES inside setup()
        'extract':  lambda X: np.sqrt(X[0]**2 + X[1]**2),   # horizontal offset
        'x_final':  0.0,             # want to stay at origin
        'step_t':   5.0,
        'ylabel':   'horizontal offset  [m]',
        'metric':   'ss_error',
        # override dict patched in setup() to add the actual DISTURBANCES entry
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
        'metric':   'rise_time',
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

    rt  = _rise_time(t, sig, x_init, x_final)
    st  = _settling_time(t, sig, x_final)
    os  = _overshoot(sig, x_init, x_final)
    sse = _ss_error(t, sig, x_final)

    return {
        'rise_time':    rt,
        'settling_time': st,
        'overshoot_pct': os,
        'ss_error':      sse,
        'x_init':        x_init,
        'signal':        sig,
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
    ov = dict(cfg['overrides'])
    # Patch disturbance test constant force
    if 'dist_F' in cfg:
        ov['gust_force'] = cfg['dist_F']
        ov.pop('traj_segments', None)
    # Remove None values
    ov = {k: v for k, v in ov.items() if v is not None}
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
    sig     = metrics['signal']

    print(f"  Rise time (10→90%):  {metrics['rise_time']:.3f} s"
          if not np.isnan(metrics['rise_time']) else "  Rise time: N/A")
    print(f"  Settling time (2%):  {metrics['settling_time']:.3f} s"
          if not np.isnan(metrics['settling_time']) else "  Settling time: N/A")
    print(f"  Overshoot:           {metrics['overshoot_pct']:.1f} %")
    print(f"  Steady-state error:  {metrics['ss_error']:.4f} m")
    print(f"  Crashed:             {result['crashed']}")

    # ── Time response plot ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, sig, color='steelblue', lw=1.8, label='Actual')
    ax.axhline(cfg['x_final'],  color='red',    lw=1.2, ls='--', label='Reference')
    ax.axhline(metrics['x_init'], color='gray', lw=1.0, ls=':',  label='Initial')
    ax.axvline(cfg['step_t'],   color='orange', lw=1.0, ls=':',  label='Step at t')
    # ±2% settling band
    amp = abs(cfg['x_final'] - metrics['x_init'])
    if amp > 1e-6:
        ax.axhspan(cfg['x_final'] - 0.02*amp, cfg['x_final'] + 0.02*amp,
                   color='green', alpha=0.12, label='±2 % settle band')
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel(cfg['ylabel'])
    ax.set_title(f"{cfg['label']}  —  {ekf_str}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out = f"response_{vehicle}_{test}.png"
    fig.savefig(out, dpi=150)
    plt.show()
    print(f"\n  Plot saved → {out}")


def sweep(vehicle, test, ekf_flag):
    """Tornado sweep: vary one parameter, track primary metric."""
    key = (test, vehicle)
    if key not in TEST_CONFIGS:
        print(f"No test config for ({test}, {vehicle})")
        return

    cfg    = TEST_CONFIGS[key]
    run_fn, lbl = _load(vehicle)
    primary = cfg['metric']

    print(f"\n{'='*60}")
    print(f"  TORNADO SWEEP — {cfg['label']}  [{lbl}]")
    print(f"  Primary metric: {primary}")
    print(f"{'='*60}")

    # Nominal run
    ov_nom  = _build_overrides(cfg, ekf_flag)
    nom_res = _run(run_fn, ov_nom, timeseries=True)
    if 't' in nom_res:
        nom_m = _compute_metrics(nom_res['t'], nom_res['X'], cfg)
        nom_val = nom_m[primary]
    else:
        nom_val = nom_res.get('rms_pos_error', np.nan)
    print(f"  Nominal {primary}: {nom_val:.4f}")

    sensitivities = []
    for param_label, key_p, values in SWEEP_PARAMS:
        vals = []
        print(f"\n  {param_label}:")
        for v in values:
            ov = _build_overrides(cfg, ekf_flag, {key_p: v})
            res = _run(run_fn, ov, timeseries=True)
            if 't' in res:
                m = _compute_metrics(res['t'], res['X'], cfg)
                val = m[primary]
            else:
                val = np.nan
            crash = "  CRASH" if res['crashed'] else ""
            print(f"    {key_p}={v:.3f}  {primary}={val:.4f}{crash}")
            vals.append(val if not np.isnan(val) else nom_val * 2)
        deltas = np.array(vals) - (nom_val if not np.isnan(nom_val) else 0)
        sensitivities.append((param_label, deltas.min(), deltas.max()))

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
    out = f"tornado_{vehicle}_{test}.png"
    fig.savefig(out, dpi=150)
    plt.show()
    print(f"\n  Tornado saved → {out}")


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
        print(f"  Run {i+1:3d}/{MC_RUNS}  m×{mass_f[i]:.2f}  "
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
    parser = argparse.ArgumentParser(
        description="Quadcopter controller robustness analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--vehicle',  choices=['sc', 'uw', 'both'], default='sc',
                        help='Vehicle to test (default: sc)')
    parser.add_argument('--test',     choices=['step_z', 'step_xy', 'turn', 'disturbance'],
                        required=True,
                        help='Test type')
    parser.add_argument('--analysis', choices=['single', 'sweep', 'mc'], default='single',
                        help='Analysis mode (default: single)')
    ekf_grp = parser.add_mutually_exclusive_group()
    ekf_grp.add_argument('--ekf',    dest='ekf', action='store_true',  default=None,
                         help='Force EKF on')
    ekf_grp.add_argument('--no-ekf', dest='ekf', action='store_false',
                         help='Force EKF off')
    args = parser.parse_args()

    ekf_flag = args.ekf   # True | False | None
    vehicles = ['sc', 'uw'] if args.vehicle == 'both' else [args.vehicle]

    for veh in vehicles:
        if args.analysis == 'single':
            single(veh, args.test, ekf_flag)
        elif args.analysis == 'sweep':
            sweep(veh, args.test, ekf_flag)
        elif args.analysis == 'mc':
            mc(veh, args.test, ekf_flag)


if __name__ == '__main__':
    main()
