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

# options: aerial, aquatic, final

OPTIONS_AERIAL = [
    "Fixed wing VTOL",
    "Moving wing VTOL",
    "Bicopter",
    "Tricopter",
    "Quad+ copter",
]

OPTIONS_AQUATIC = [
    "Biomimetics",
    "Hydrojet",
    "Propellers",
    "Voith-Schneider",
]

OPTIONS_FINAL = [
    "Quad+ and Prop",
    "Quad+ and VS",
    "Quad+ and H-jet",
    "Quad+ and Bio",
    "Tri and Prop",
    "Bi and Prop",
]

# criteria: aerial, aquatic, final

CRITERIA_AERIAL = {
    "Performance": {
        "weight": 40,
        "sub": {
            "Performance": (100, [1, 3, 2, 2, 3])},
    },
    "Cost": {
        "weight": 25,
        "sub": {
            "Cost": (100, [2, 1, 2, 2, 2])},
    },
    "Risk": {
        "weight": 20,
        "sub": {
            "Risk": (100, [2, 1, 2, 2, 3])},
    },
    "Mass": {
        "weight": 15,
        "sub": {
            "Mass": (100, [2, 2, 3, 3, 3])},
    },
}

CRITERIA_AQUATIC = {
    "Performance": {
        "weight": 40,
        "sub": {
            "Performance": (100, [3, 2, 2, 3])},
    },
    "Cost": {
        "weight": 25,
        "sub": {
            "Cost": (100, [1, 2, 3, 2])},
    },
    "Risk": {
        "weight": 20,
        "sub": {
            "Risk": (100, [1, 2, 2, 1])},
    },
    "Mass": {
        "weight": 15,
        "sub": {
            "Mass": (100, [1, 1, 2, 2])},
    },
}

CRITERIA_FINAL = {
    "Performance": {
        "weight": 40,
        "sub": {
            "Mass":                       (20, [3, 2, 2, 1, 3, 3]),
            "Aerial manoeuvrability":     (10, [3, 3, 3, 2, 2, 1]),
            "Underwater manoeuvrability": (10, [2, 2, 1, 3, 2, 2]),
            "Damage tolerance":           (10, [2, 2, 2, 3, 1, 1]),
            "Aerial efficiency":          (15, [2, 2, 2, 2, 3, 3]),
            "Underwater efficiency":      (10, [2, 3, 1, 3, 2, 2]),
            "Payload":                    ( 5, [3, 2, 1, 1, 3, 3]),
            "Transition performance":     ( 5, [2, 2, 1, 2, 2, 2]),
            "Stability":                  (15, [3, 3, 2, 3, 2, 1]),
        },
    },
    "Cost": {
        "weight": 30,
        "sub": {
            "Capex":       (40, [3, 2, 1, 1, 2, 2]),
            "Opex":        (60, [3, 3, 2, 1, 3, 3]),
        },
    },
    "Risk": {
        "weight": 20,
        "sub": {
            "Complexity":    (40, [3, 2, 2, 1, 2, 2]),
            "Project risk":  (30, [3, 2, 1, 1, 3, 3]),
            "Manufacturing": (30, [3, 3, 2, 1, 3, 3]),
        },
    },
    "Sustainability": {
        "weight": 10,
        "sub": {
            "Noise":              (50, [1, 2, 1, 2, 3, 3]),
            "Marine life impact": (50, [2, 2, 3, 3, 2, 2]),
        },
    },
}


# criteria colors setting

COLORS_AERIAL = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']
COLORS_AQUATIC = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']
COLORS_FINAL = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3', "#a65628"] # "#17becf"

CRIT_COLORS_AERIAL = {'Performance': "#df80d4", 'Cost': "#85bbdb",
               'Risk': "#8fc262", 'Mass': "#dbdd57"}
CRIT_COLORS_AQUATIC = CRIT_COLORS_AERIAL
CRIT_COLORS_FINAL = {'Performance': "#df80d4", 'Cost': "#85bbdb",
               'Risk': "#8fc262", 'Sustainability': "#dbdd57"}

# select for which trade-off you want to perform the sensitivity analysis

OPTIONS = OPTIONS_AQUATIC # choose: OPTIONS_AERIAL, OPTIONS_AQUATIC, OPTIONS_FINAL
CRITERIA = CRITERIA_AQUATIC # choose: CRITERIA_AERIAL, CRITERIA_AQUATIC, CRITERIA_FINAL
COLORS = COLORS_AQUATIC # choose: COLORS_AERIAL, COLORS_AQUATIC, COLORS_FINAL
CRIT_COLORS = CRIT_COLORS_AQUATIC # choose: CRIT_COLORS_AERIAL, CRIT_COLORS_AQUATIC, CRIT_COLORS_FINAL

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
                ax.axvspan(w_range[j], w_range[j+1], color='yellow', alpha=0.6)

        offset_map = {}

        for i, (opt, col) in enumerate(zip(OPTIONS, COLORS)):

            lw = 2.8 if i == winner_idx else 1.6
            alpha = 1.0 if i == winner_idx else 0.75

            # Detect overlapping curves
            key = tuple(np.round(matrix[i], 6))

            if key in offset_map:
                offset = offset_map[key]
                offset_map[key] += 0.005
                ls = '--'
            else:
                offset_map[key] = 0.01
                offset = 0.0
                ls = '-'

            ax.plot(
                w_range,
                matrix[i] + offset,
                color=col,
                lw=lw,
                ls=ls,
                alpha=alpha,
                label=opt
            )

        #ax.axvline(base_w, color='black', lw=1.2, ls=':', label='Baseline')
        ax.set_title(f'{crit}  (baseline = {base_w}%)', fontweight='bold')
        ax.set_xlabel(f'{crit} weight (%)')
        ax.set_ylabel('Total score')
        ax.set_xlim(w_range[0], w_range[-1])
        ax.grid(True, alpha=0.25)

    handles = [plt.Line2D([0],[0], color=c, lw=2, label=o)
               for o, c in zip(OPTIONS, COLORS)]
    handles += [mpatches.Patch(color='yellow', label='Other option scores highest')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'Sensitivity Analysis - Criterion Weight Variation (±{WEIGHT_MARGIN}%)',
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

    fig, ax = plt.subplots(figsize=(14, 5)) # 14 was 12
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
    ax.set_title('Sensitivity Analysis - Criterion Elimination\n'
                 '(eliminated criterion weight redistributed proportionally)',
                 fontweight='bold')
    #ax.legend(ncol=3, fontsize=9, loc='upper right')

    ax.legend(
    ncol=1,
    fontsize=9,
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0
    )

    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    #plt.savefig('sensitivity_elimination.png', dpi=150, bbox_inches='tight')
    plt.show()

# # ── Figure 3: Score risk map ──────────────────────────────────────────────────

# def plot_score_risk():
#     """
#     Scatter plot: x = weight impact of sub-criterion (how much 1 score point moves total),
#                   y = score gap between winner and runner-up on that sub-criterion.
#     Bottom-right = high impact, small gap → most at risk of changing the outcome.
#     """
#     baseline   = calculate()
#     sorted_idx = sorted(range(len(OPTIONS)), key=lambda i: baseline[i], reverse=True)
#     winner     = sorted_idx[0]
#     runner_up  = sorted_idx[1]

#     impacts, gaps, labels, colors = [], [], [], []
#     for crit, data in CRITERIA.items():
#         W = data["weight"] / 100
#         for sub_name, (w, raw) in data["sub"].items():
#             impact = W * (w / 100)           # score-point → total-score sensitivity
#             gap    = raw[winner] - raw[runner_up]   # positive = winner leads here
#             impacts.append(impact * 100)     # express as % of total score per point
#             gaps.append(gap)
#             labels.append(sub_name)
#             colors.append(CRIT_COLORS[crit])

#     fig, ax = plt.subplots(figsize=(12, 8))
#     texts = []
#     for x, y, lbl, col in zip(impacts, gaps, labels, colors):
#         ax.scatter(x, y, color=col, s=120, zorder=3, edgecolors='white', linewidth=0.5)
#         texts.append(ax.text(x, y, lbl, fontsize=8))
#     adjust_text(texts, ax=ax,
#                 arrowprops=dict(arrowstyle='-', color='gray', lw=0.6),
#                 expand=(1.3, 1.5), force_text=(0.5, 0.8))

#     ax.axhline(0, color='gray', lw=0.8, ls='--')
#     ax.axvline(np.percentile(impacts, 66), color='red', lw=0.8, ls=':',
#                label='Top-33% impact threshold')

#     # quadrant labels
#     ax.text(0.98, 0.98, 'Winner leads,\nhigh impact', transform=ax.transAxes,
#             ha='right', va='top', color='green', fontsize=8, alpha=0.7)
#     ax.text(0.98, 0.02, 'Runner-up leads,\nhigh impact', transform=ax.transAxes,
#             ha='right', va='bottom', color='red', fontsize=8, alpha=0.7)

#     # legend for criteria colours
#     legend_patches = [mpatches.Patch(color=c, label=k)
#                       for k, c in CRIT_COLORS.items()]
#     ax.legend(handles=legend_patches + [
#         plt.Line2D([0],[0], color='red', ls=':', lw=1, label='Top-33% impact')],
#               fontsize=9, loc='upper left')

#     ax.set_xlabel(f'Weight impact  (total-score change per ±{SCORE_MARGIN} score point,  %)')
#     ax.set_ylabel(f'Score gap  (winner − runner-up)\n'
#                   f'[{OPTIONS[winner]} vs {OPTIONS[runner_up]}]')
#     ax.set_title('Score Risk Map — Sub-criterion Sensitivity\n'
#                  'Bottom-right: high impact AND small/negative gap → most vulnerable',
#                  fontweight='bold')
#     ax.grid(True, alpha=0.2)
#     plt.tight_layout()
#     #plt.savefig('sensitivity_score_risk.png', dpi=150, bbox_inches='tight')
#     plt.show()

# ── Figure 4: Individual score perturbation ────────────────────────────────

def _calculate_with_single_score_delta(
        crit_name,
        sub_name,
        option_idx,
        delta):
    """
    Recompute totals after perturbing ONE score for ONE option
    on ONE sub-criterion by delta (clamped 0–3).
    """

    n = len(OPTIONS)
    totals = [0.0] * n

    for crit, data in CRITERIA.items():

        W = data["weight"] / 100

        for sname, (w, raw) in data["sub"].items():

            scores = raw.copy()

            if crit == crit_name and sname == sub_name:
                scores[option_idx] = min(
                    3,
                    max(0, scores[option_idx] + delta)
                )

            for i in range(n):
                totals[i] += W * (w / 100) * scores[i]

    return totals


def plot_score_perturbation():

    baseline = calculate()

    sorted_idx = sorted(
        range(len(OPTIONS)),
        key=lambda i: baseline[i],
        reverse=True
    )

    winner = sorted_idx[0]
    runner_up = sorted_idx[1]

    # robustness bounds
    lo = baseline.copy()
    hi = baseline.copy()

    flips = []

    # store vulnerability metrics
    vuln_labels = []
    vuln_values = []
    vuln_colors = []

    # ──────────────────────────────────────────────────────────────────────
    # Run perturbations
    # ──────────────────────────────────────────────────────────────────────

    for crit, data in CRITERIA.items():
        for sub_name, (w, raw) in data["sub"].items():
            for option_idx in range(len(OPTIONS)):
                for delta in (+SCORE_MARGIN, -SCORE_MARGIN):

                    new_totals = _calculate_with_single_score_delta(
                        crit,
                        sub_name,
                        option_idx,
                        delta
                    )

                    # update robustness bounds
                    for i in range(len(OPTIONS)):
                        lo[i] = min(lo[i], new_totals[i])
                        hi[i] = max(hi[i], new_totals[i])

                    new_winner = np.argmax(new_totals)

                    # record ranking flips
                    if new_winner != winner:

                        flips.append({
                            'criterion': crit,
                            'sub': sub_name,
                            'option': OPTIONS[option_idx],
                            'delta': delta,
                            'new_winner': OPTIONS[new_winner],
                            'totals': new_totals
                        })

                    # vulnerability metric:
                    # reduction in winner advantage
                    baseline_gap = (baseline[winner] - baseline[runner_up])

                    new_gap = (new_totals[winner] - new_totals[runner_up])

                    vulnerability = baseline_gap - new_gap

                    vuln_labels.append(
                        f"{sub_name}\n({OPTIONS[option_idx]} {delta:+d})"
                    )

                    vuln_values.append(vulnerability)
                    vuln_colors.append(CRIT_COLORS[crit])

    # ──────────────────────────────────────────────────────────────────────
    # Print flips
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("INDIVIDUAL SCORE PERTURBATION — RANKING FLIPS")
    print("=" * 70)

    if not flips:
        print("No single ±1 score perturbation changes the winner.")

    else:

        for f in flips:

            print(
                f"[{f['criterion']}] "
                f"{f['sub']} | "
                f"{f['option']} ({f['delta']:+d}) "
                f"-> new winner: {f['new_winner']}"
            )

    # ──────────────────────────────────────────────────────────────────────
    # Plotting
    # ──────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        constrained_layout=True
    )

    # =====================================================================
    # LEFT PLOT — robustness ranges
    # =====================================================================

    ax = axes[0]

    x = np.arange(len(OPTIONS))

    for i, (opt, col) in enumerate(zip(OPTIONS, COLORS)):

        ax.bar(
            x[i],
            baseline[i],
            color=col,
            alpha=0.85,
            zorder=3,
            label=opt
        )

        center = (hi[i] + lo[i]) / 2

        ax.errorbar(
            x[i],
            center,
            yerr=[
                [center - lo[i]],
                [hi[i] - center]
            ],
            fmt='none',
            color='black',
            capsize=6,
            lw=2,
            zorder=4
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [o.replace(' ', '\n') for o in OPTIONS],
        fontsize=9
    )

    ax.set_ylabel('Total score')

    ax.set_ylim(0, 3)

    ax.set_title(
        'Robustness range under all\nsingle-score perturbations',
        fontweight='bold'
    )

    ax.grid(True, axis='y', alpha=0.25)

    ax.legend(
        title='Design option', 
        fontsize=8, 
        title_fontsize=9,
        loc='lower right'
    )

    # highlight overlap region
    ax.axhspan(
        lo[winner],
        hi[runner_up],
        color='red',
        alpha=0.08,
        zorder=0
    )

    # =====================================================================
    # RIGHT PLOT — vulnerability ranking
    # =====================================================================

    ax2 = axes[1]
    order = np.argsort(vuln_values)[::-1]
    top_n = 15
    order = order[:top_n]
    y = np.arange(len(order))

    ax2.barh(
        y,
        [vuln_values[i] for i in order],
        color=[vuln_colors[i] for i in order],
        alpha=0.85
    )

    ax2.set_yticks(y)

    ax2.set_yticklabels(
        [vuln_labels[i] for i in order],
        fontsize=8
    )

    ax2.invert_yaxis()

    ax2.set_xlabel(
        'Reduction in winner advantage'
    )

    ax2.set_title(
        'Most critical individual scores',
        fontweight='bold'
    )

    ax2.grid(True, axis='x', alpha=0.25)

    # legend for criterion colours
    legend_patches = [
        mpatches.Patch(color=color, label=crit)
        for crit, color in CRIT_COLORS.items()
    ]

    ax2.legend(
        handles=legend_patches,
        title='Criterion',
        fontsize=8,
        title_fontsize=9,
        loc='lower right'
    )

    fig.suptitle(
        f'Sensitivity Analysis - Individual Score Variation (±{SCORE_MARGIN})',
        fontsize=14,
        fontweight='bold'
    )

    plt.show()       


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_report()
    plot_weight_sweep()
    plot_elimination()
    #plot_score_risk()
    plot_score_perturbation()