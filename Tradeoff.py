"""
Trade-off matrix — 5 design options
Options: 1=Fixed wing VTOL, 2=Moving wing VTOL, 3=Rotary wing,
         4=Biomimetics, 5=Detachable

Scores:  0=Unfeasible, 1=Fixable deficiencies,
         2=Meets requirements, 3=Exceeds requirements

Total score per option = Σ_c (W_c/100) * Σ_s (w_s/100 * score_s)   →  0–3 scale
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text

# ── Sensitivity margins (change these to explore different assumptions) ──────
WEIGHT_MARGIN = 10   # ±pp variation on each criterion weight
#   Justification: reflects the typical spread in team member weight preferences;
#   10 pp is also a conventional starting point for first-order robustness checks.
SCORE_MARGIN  = 1    # ±1 score perturbation per sub-criterion
#   Justification: scores are discrete integers 0–3; ±1 is the smallest
#   meaningful perturbation and reflects realistic scoring disagreement.
# ─────────────────────────────────────────────────────────────────────────────

OPTIONS = [
    "Fixed wing VTOL",
    "Moving wing VTOL",
    "Rotary wing",
    "Biomimetics",
    "Detachable",
]

CRITERIA = {
    "Performance": {
        "weight": 40,
        "sub": {
            "Mass":                       (15, [2, 2, 3, 1, 1]),
            "Aerial manoeuvrability":     (10, [1, 2, 3, 2, 3]),
            "Underwater manoeuvrability": (10, [1, 2, 3, 3, 3]),
            "Endurance":                  (15, [3, 3, 2, 2, 1]),
            "Damage tolerance":           (10, [2, 2, 2, 2, 3]),
            "Aerial efficiency":          (10, [2, 2, 3, 2, 3]),
            "Underwater efficiency":      ( 5, [1, 2, 2, 3, 3]),
            "Payload":                    ( 5, [2, 2, 2, 2, 1]),
            "Transition performance":     ( 5, [1, 3, 3, 3, 3]),
            "Stability":                  (15, [2, 3, 1, 3, 3]),
        },
    },
    "Cost": {
        "weight": 30,
        "sub": {
            "Capex":       (25, [3, 2, 2, 1, 1]),
            "Opex":        (45, [1, 2, 2, 2, 1]),
            "Maintenance": (30, [2, 2, 2, 1, 1]),
        },
    },
    "Risk": {
        "weight": 20,
        "sub": {
            "Complexity":    (40, [2, 2, 3, 1, 3]),
            "Project risk":  (30, [2, 2, 2, 1, 3]),
            "Manufacturing": (30, [2, 2, 2, 2, 3]),
        },
    },
    "Sustainability": {
        "weight": 10,
        "sub": {
            "Noise":              (50, [1, 2, 2, 3, 2]),
            "Marine life impact": (50, [2, 2, 2, 3, 2]),
        },
    },
}

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']
CRIT_COLORS = {'Performance': '#1f77b4', 'Cost': '#ff7f0e',
               'Risk': '#2ca02c', 'Sustainability': '#d62728'}

# ── Core calculation ─────────────────────────────────────────────────────────

def _sub_scores(crit_data):
    """Return per-option weighted sub-scores for one criterion (not yet * W_c)."""
    n = len(OPTIONS)
    scores = [0.0] * n
    for _, (w, raw) in crit_data["sub"].items():
        for i in range(n):
            scores[i] += (w / 100) * raw[i]
    return scores


def calculate(weight_overrides=None):
    """
    Return list of total scores, one per option.
    weight_overrides: dict {criterion: weight_percent} – defaults to CRITERIA weights.
    """
    if weight_overrides is None:
        weight_overrides = {c: CRITERIA[c]["weight"] for c in CRITERIA}
    totals = [0.0] * len(OPTIONS)
    for crit, data in CRITERIA.items():
        W = weight_overrides[crit] / 100
        for i, s in enumerate(_sub_scores(data)):
            totals[i] += W * s
    return totals


def _redistribute(varied_crit, new_w):
    """Set varied_crit to new_w; redistribute remainder proportionally to others."""
    base = {c: CRITERIA[c]["weight"] for c in CRITERIA}
    others_total = sum(w for c, w in base.items() if c != varied_crit)
    remaining = 100.0 - new_w
    result = {varied_crit: new_w}
    for c, w in base.items():
        if c != varied_crit:
            result[c] = remaining * w / others_total
    return result

# ── Print report ─────────────────────────────────────────────────────────────

def print_report():
    totals = calculate()
    col_w = 22
    header = f"{'Criterion':<20}" + "".join(f"{o:>{col_w}}" for o in OPTIONS)
    sep = "=" * len(header)
    print(sep)
    print("TRADE-OFF RESULTS  (score 0–3 scale)")
    print(sep)
    print(header)
    print("-" * len(header))
    for crit, data in CRITERIA.items():
        sub   = _sub_scores(data)
        W     = data["weight"] / 100
        label = crit + " (w=" + str(data["weight"]) + "%)"
        row   = f"{label:<20}" + "".join(f"{W*s:>{col_w}.3f}" for s in sub)
        print(row)
    print("-" * len(header))
    print(f"{'TOTAL':<20}" + "".join(f"{t:>{col_w}.3f}" for t in totals))
    print(sep)
    ranked = sorted(enumerate(totals), key=lambda x: x[1], reverse=True)
    print("\nRANKING:")
    for rank, (i, score) in enumerate(ranked, 1):
        print(f"  {rank}. {OPTIONS[i]:<25}  {score:.3f}")

# ── Figure 1: Weight sweep ────────────────────────────────────────────────────

def plot_weight_sweep():
    crits = list(CRITERIA.keys())
    baseline = calculate()
    winner_idx = baseline.index(max(baseline))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, crit in zip(axes, crits):
        base_w  = CRITERIA[crit]["weight"]
        w_range = np.linspace(max(1, base_w - WEIGHT_MARGIN),
                              min(99, base_w + WEIGHT_MARGIN), 200)

        matrix = np.zeros((len(OPTIONS), len(w_range)))
        for j, w in enumerate(w_range):
            s = calculate(_redistribute(crit, w))
            for i in range(len(OPTIONS)):
                matrix[i, j] = s[i]

        # shade regions where the winner changes
        current_winner = np.argmax(matrix, axis=0)
        for j in range(len(w_range) - 1):
            if current_winner[j] != winner_idx:
                ax.axvspan(w_range[j], w_range[j+1], color='lightyellow', alpha=0.6)

        for i, (opt, col) in enumerate(zip(OPTIONS, COLORS)):
            lw = 2.5 if i == winner_idx else 1.2
            ls = '-'  if i == winner_idx else '--'
            ax.plot(w_range, matrix[i], color=col, lw=lw, ls=ls, label=opt)

        ax.axvline(base_w, color='black', lw=1.2, ls=':', label='Baseline')
        ax.set_title(f'{crit}  (baseline = {base_w}%)', fontweight='bold')
        ax.set_xlabel(f'{crit} weight (%)')
        ax.set_ylabel('Total score')
        ax.set_xlim(w_range[0], w_range[-1])
        ax.grid(True, alpha=0.25)

    handles = [plt.Line2D([0],[0], color=c, lw=2, label=o)
               for o, c in zip(OPTIONS, COLORS)]
    handles += [plt.Line2D([0],[0], color='black', lw=1.2, ls=':', label='Baseline'),
                mpatches.Patch(color='lightyellow', label='Rank change region')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'Sensitivity Analysis — Criterion Weight Variation  (±{WEIGHT_MARGIN} pp)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    #plt.savefig('sensitivity_weights.png', dpi=150, bbox_inches='tight')
    plt.show()

# ── Figure 2: Criterion elimination ──────────────────────────────────────────

def plot_elimination():
    baseline = calculate()

    scenarios      = {'Baseline': baseline}
    scenario_labels = ['Baseline']
    for crit in CRITERIA:
        others_total = sum(CRITERIA[c]["weight"] for c in CRITERIA if c != crit)
        overrides = {c: (CRITERIA[c]["weight"] / others_total * 100 if c != crit else 0.0)
                     for c in CRITERIA}
        scenarios[f'No\n{crit}'] = calculate(overrides)
        scenario_labels.append(f'No\n{crit}')

    fig, ax = plt.subplots(figsize=(12, 5))
    n_sc  = len(scenarios)
    n_opt = len(OPTIONS)
    x     = np.arange(n_sc)
    width = 0.14
    offsets = (np.arange(n_opt) - (n_opt - 1) / 2) * width

    for i, (opt, col) in enumerate(zip(OPTIONS, COLORS)):
        vals = [s[i] for s in scenarios.values()]
        ax.bar(x + offsets[i], vals, width, label=opt, color=col, alpha=0.85,
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=10)
    ax.axvline(0.5, color='gray', ls=':', lw=1)
    ax.set_ylabel('Total score')
    ax.set_title('Criterion Elimination Sensitivity\n'
                 '(eliminated criterion weight redistributed proportionally)',
                 fontweight='bold')
    ax.legend(ncol=5, fontsize=9, loc='lower right')
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    #plt.savefig('sensitivity_elimination.png', dpi=150, bbox_inches='tight')
    plt.show()

# ── Figure 3: Score risk map ──────────────────────────────────────────────────

def plot_score_risk():
    """
    Scatter plot: x = weight impact of sub-criterion (how much 1 score point moves total),
                  y = score gap between winner and runner-up on that sub-criterion.
    Bottom-right = high impact, small gap → most at risk of changing the outcome.
    """
    baseline   = calculate()
    sorted_idx = sorted(range(len(OPTIONS)), key=lambda i: baseline[i], reverse=True)
    winner     = sorted_idx[0]
    runner_up  = sorted_idx[1]

    impacts, gaps, labels, colors = [], [], [], []
    for crit, data in CRITERIA.items():
        W = data["weight"] / 100
        for sub_name, (w, raw) in data["sub"].items():
            impact = W * (w / 100)           # score-point → total-score sensitivity
            gap    = raw[winner] - raw[runner_up]   # positive = winner leads here
            impacts.append(impact * 100)     # express as % of total score per point
            gaps.append(gap)
            labels.append(sub_name)
            colors.append(CRIT_COLORS[crit])

    fig, ax = plt.subplots(figsize=(12, 8))
    texts = []
    for x, y, lbl, col in zip(impacts, gaps, labels, colors):
        ax.scatter(x, y, color=col, s=120, zorder=3, edgecolors='white', linewidth=0.5)
        texts.append(ax.text(x, y, lbl, fontsize=8))
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.6),
                expand=(1.3, 1.5), force_text=(0.5, 0.8))

    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axvline(np.percentile(impacts, 66), color='red', lw=0.8, ls=':',
               label='Top-33% impact threshold')

    # quadrant labels
    ax.text(0.98, 0.98, 'Winner leads,\nhigh impact', transform=ax.transAxes,
            ha='right', va='top', color='green', fontsize=8, alpha=0.7)
    ax.text(0.98, 0.02, 'Runner-up leads,\nhigh impact', transform=ax.transAxes,
            ha='right', va='bottom', color='red', fontsize=8, alpha=0.7)

    # legend for criteria colours
    legend_patches = [mpatches.Patch(color=c, label=k)
                      for k, c in CRIT_COLORS.items()]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0],[0], color='red', ls=':', lw=1, label='Top-33% impact')],
              fontsize=9, loc='upper left')

    ax.set_xlabel(f'Weight impact  (total-score change per ±{SCORE_MARGIN} score point,  %)')
    ax.set_ylabel(f'Score gap  (winner − runner-up)\n'
                  f'[{OPTIONS[winner]} vs {OPTIONS[runner_up]}]')
    ax.set_title('Score Risk Map — Sub-criterion Sensitivity\n'
                 'Bottom-right: high impact AND small/negative gap → most vulnerable',
                 fontweight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    #plt.savefig('sensitivity_score_risk.png', dpi=150, bbox_inches='tight')
    plt.show()

# ── Figure 4: Score perturbation ─────────────────────────────────────────────

def _calculate_with_score_delta(crit_name, sub_name, delta):
    """Recompute totals with one sub-criterion's scores shifted by delta (clamped 0-3)."""
    n = len(OPTIONS)
    totals = [0.0] * n
    for crit, data in CRITERIA.items():
        W = data["weight"] / 100
        for sname, (w, raw) in data["sub"].items():
            if crit == crit_name and sname == sub_name:
                scores = [min(3, max(0, r + delta)) for r in raw]
            else:
                scores = raw
            for i in range(n):
                totals[i] += W * (w / 100) * scores[i]
    return totals


def plot_score_perturbation():
    """
    For each sub-criterion, shift all scores by +SCORE_MARGIN and -SCORE_MARGIN,
    recompute totals, and track the min/max range reached per option.
    Plot as a range band around the baseline — if the winner's lower bound
    overlaps the runner-up's upper bound, the result is not robust.
    Also print which individual perturbations flip the ranking.
    """
    baseline   = calculate()
    sorted_idx = sorted(range(len(OPTIONS)), key=lambda i: baseline[i], reverse=True)
    winner     = sorted_idx[0]
    runner_up  = sorted_idx[1]

    # Accumulate min/max per option across all single sub-criterion perturbations
    lo = baseline[:]
    hi = baseline[:]
    flips = []   # (sub-criterion label, delta, new totals)

    for crit, data in CRITERIA.items():
        for sub_name in data["sub"]:
            for delta in (+SCORE_MARGIN, -SCORE_MARGIN):
                new_totals = _calculate_with_score_delta(crit, sub_name, delta)
                for i in range(len(OPTIONS)):
                    lo[i] = min(lo[i], new_totals[i])
                    hi[i] = max(hi[i], new_totals[i])
                new_winner = new_totals.index(max(new_totals))
                if new_winner != winner:
                    flips.append((crit, sub_name, delta, new_totals))

    # ── Print flip summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCORE PERTURBATION — RANKING FLIPS")
    print("=" * 60)
    if not flips:
        print("No single ±1 score change flips the winner.")
    else:
        for crit, sub_name, delta, tots in flips:
            new_w = tots.index(max(tots))
            sign  = "+" if delta > 0 else ""
            print(f"  [{crit}] {sub_name}  ({sign}{delta})  "
                  f"-> new winner: {OPTIONS[new_w]}  "
                  f"(was {OPTIONS[winner]})")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: range band per option
    ax = axes[0]
    x  = np.arange(len(OPTIONS))
    for i, (opt, col) in enumerate(zip(OPTIONS, COLORS)):
        ax.bar(x[i], baseline[i], color=col, alpha=0.85, label=opt, zorder=3)
        ax.errorbar(x[i], (hi[i] + lo[i]) / 2,
                    yerr=[[(hi[i] + lo[i]) / 2 - lo[i]],
                          [hi[i] - (hi[i] + lo[i]) / 2]],
                    fmt='none', color='black', capsize=6, lw=2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([o.replace(' ', '\n') for o in OPTIONS], fontsize=8)
    ax.set_ylabel('Total score')
    ax.set_title(f'Score range across all single\nsub-criterion ±{SCORE_MARGIN} perturbations',
                 fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(fontsize=8)

    # Right: per-sub-criterion delta on winner vs runner-up
    ax2 = axes[1]
    sub_labels, w_deltas, ru_deltas, bar_colors = [], [], [], []

    for crit, data in CRITERIA.items():
        for sub_name in data["sub"]:
            down = _calculate_with_score_delta(crit, sub_name, -SCORE_MARGIN)
            up   = _calculate_with_score_delta(crit, sub_name, +SCORE_MARGIN)
            w_deltas.append(down[winner]   - baseline[winner])    # pessimistic for winner
            ru_deltas.append(up[runner_up] - baseline[runner_up]) # optimistic for runner-up
            sub_labels.append(f"{crit[:4]}. {sub_name}")
            bar_colors.append(CRIT_COLORS[crit])

    y = np.arange(len(sub_labels))
    ax2.barh(y - 0.2, w_deltas,  0.35, color=COLORS[winner],   alpha=0.8,
             label=f'{OPTIONS[winner]} (score -1)')
    ax2.barh(y + 0.2, ru_deltas, 0.35, color=COLORS[runner_up], alpha=0.8,
             label=f'{OPTIONS[runner_up]} (score +1)')
    ax2.axvline(0, color='black', lw=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(sub_labels, fontsize=7.5)
    ax2.set_xlabel('Change in total score')
    ax2.set_title(f'Worst case for winner vs best case\nfor runner-up, per sub-criterion',
                  fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, axis='x', alpha=0.25)

    fig.suptitle(f'Score Sensitivity — Actual ±{SCORE_MARGIN} Score Perturbations',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    #plt.savefig('sensitivity_score_perturb.png', dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_report()
    plot_weight_sweep()
    plot_elimination()
    plot_score_risk()
    plot_score_perturbation()
