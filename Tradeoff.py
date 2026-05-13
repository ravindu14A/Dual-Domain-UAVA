import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 0. Global Parameters
# ==========================================
# Defines the +/- percentage sweep range for the visual line graphs (e.g., 20 means +/- 20%)
SENSITIVITY_SWEEP_PERCENT = 20

# ==========================================
# 1. Binary Compatibility Matrix (from whiteboard)
# ==========================================
propulsion_types = ["Hydrojet", "Propellers", "Voith-Schneider prop", "Biomimetics"]
vehicle_types = ["Fixed wing VTOL", "Moving wing VTOL", "Bicopter", "Tricopter", "Quad+ copter"]

binary_data = [
    [1, 1, 1, 1, 1],  # Hydrojet
    [1, 1, 1, 1, 1],  # Propellers
    [0, 0, 1, 1, 1],  # Voith-Schneider prop
    [0, 0, 0, 1, 1]   # Biomimetics
]

binary_compatibility_matrix = pd.DataFrame(binary_data, index=propulsion_types, columns=vehicle_types)

# ==========================================
# 2. AERIAL Concept Evaluation Matrix (Dynamic)
# ==========================================
aerial_criteria = ["Performance", "Cost", "Risk", "Mass"]
aerial_columns = ["Weight", "Fixed wing VTOL", "Moving wing VTOL", "Bicopter", "Tricopter", "Quad+ copter"]

aerial_raw_data = [
    [40, 1, 3, 2, 2, 3], # Performance
    [25, 2, 1, 2, 2, 2], # Cost
    [20, 2, 1, 2, 2, 3], # Risk
    [15, 2, 2, 3, 3, 3]  # Mass
]

aerial_matrix = pd.DataFrame(aerial_raw_data, index=aerial_criteria, columns=aerial_columns)

aerial_weights = aerial_matrix["Weight"] / 100
aerial_vehicle_cols = aerial_matrix.columns[1:]
aerial_total_scores = aerial_matrix[aerial_vehicle_cols].multiply(aerial_weights, axis=0).sum()

aerial_matrix.loc["Total score"] = pd.concat([pd.Series({"Weight": ""}), aerial_total_scores])

# ==========================================
# 3. AQUATIC Concept Evaluation Matrix (Dynamic)
# ==========================================
aquatic_criteria = ["Performance", "Cost", "Risk", "Mass"]
aquatic_columns = ["Weight", "Biomimetics", "Hydrojet", "Propellers", "Voith-Schneider prop"]

aquatic_raw_data = [
    [40, 3, 2, 2, 3], # Performance
    [25, 1, 2, 3, 2], # Cost
    [20, 1, 2, 2, 1], # Risk
    [15, 1, 1, 2, 2]  # Mass
]

aquatic_matrix = pd.DataFrame(aquatic_raw_data, index=aquatic_criteria, columns=aquatic_columns)

aquatic_weights = aquatic_matrix["Weight"] / 100
aquatic_propulsion_cols = aquatic_matrix.columns[1:]
aquatic_total_scores = aquatic_matrix[aquatic_propulsion_cols].multiply(aquatic_weights, axis=0).sum()

aquatic_matrix.loc["Total score"] = pd.concat([pd.Series({"Weight": ""}), aquatic_total_scores])

# ==========================================
# 4. Weighted Compatibility Matrix (Dynamic)
# ==========================================
weighted_compat_rows = ["Biomimetics", "Hydrojet", "Propellers", "Voith-Schneider prop"]
weighted_compat_cols = ["Fixed wing VTOL", "Moving wing VTOL", "Bicopter", "Tricopter", "Quad+ copter"]

weighted_compatibility_matrix = pd.DataFrame(index=weighted_compat_rows, columns=weighted_compat_cols)

for propulsion in weighted_compat_rows:
    for vehicle in weighted_compat_cols:
        aquatic_score = aquatic_matrix.loc["Total score", propulsion]
        aerial_score = aerial_matrix.loc["Total score", vehicle]
        weighted_compatibility_matrix.loc[propulsion, vehicle] = (aquatic_score + aerial_score) / 2

weighted_compatibility_matrix = weighted_compatibility_matrix.astype(float)

# ==========================================
# 5. Filter and Rank Compatible Combinations
# ==========================================
binary_matrix_aligned = binary_compatibility_matrix.reindex(index=weighted_compat_rows)
valid_combos = []

for propulsion in weighted_compat_rows:
    for vehicle in weighted_compat_cols:
        is_compatible = binary_matrix_aligned.loc[propulsion, vehicle]
        if is_compatible == 1:
            score = weighted_compatibility_matrix.loc[propulsion, vehicle]
            valid_combos.append({
                "Propulsion": propulsion,
                "Vehicle": vehicle,
                "Score": score
            })

results_df = pd.DataFrame(valid_combos)
ranked_results = results_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
ranked_results.index = ranked_results.index + 1
top_6_results = ranked_results.head(6)

# ==========================================
# 6. Final Trade-off Matrix (Dynamically Linked)
# ==========================================
concept_names = {}
concept_cols = []

for index, row in top_6_results.iterrows():
    name = f"{row['Vehicle']} & {row['Propulsion']}"
    str_index = str(index)
    concept_names[str_index] = name
    concept_cols.append(str_index)

main_weights = {
    "Performance": 40,
    "Cost": 30,
    "Risk": 20,
    "Sustainability": 10
}

raw_tradeoff_data = [
    ["Performance", "Mass", 20, 3, 2, 2, 1, 3, 3],
    ["Performance", "Aerial manoeuvrability", 10, 3, 3, 3, 2, 2, 1],
    ["Performance", "Underwater manoeuvrability", 10, 2, 2, 1, 3, 2, 2],
    ["Performance", "Endurance", 0, 0, 0, 0, 0, 0, 0], 
    ["Performance", "Damage tolerance", 10, 2, 2, 2, 3, 1, 1],
    ["Performance", "Aerial efficiency (hover)", 15, 2, 2, 2, 2, 3, 3],
    ["Performance", "Underwater efficiency", 10, 2, 3, 1, 3, 2, 2],
    ["Performance", "Payload", 5, 3, 2, 1, 1, 3, 3],
    ["Performance", "Transition performance", 5, 2, 2, 1, 2, 2, 2],
    ["Performance", "Stability", 15, 3, 3, 2, 3, 2, 1],

    ["Cost", "CAPEX", 40, 3, 2, 1, 1, 2, 2],
    ["Cost", "OPEX", 60, 3, 3, 2, 1, 3, 3],
    ["Cost", "Maintenance", 0, 0, 0, 0, 0, 0, 0],

    ["Risk", "Complexity", 40, 3, 2, 2, 1, 2, 2],
    ["Risk", "Project risk", 30, 3, 2, 1, 1, 3, 3],
    ["Risk", "Manufacturing", 30, 3, 3, 2, 1, 3, 3],

    ["Sustainability", "Noise", 50, 1, 2, 1, 2, 3, 3],
    ["Sustainability", "Marine life impact", 50, 2, 2, 3, 3, 2, 2]
]

tradeoff_columns = ["Main Criteria", "Subcriteria", "Weight"] + concept_cols
tradeoff_df = pd.DataFrame(raw_tradeoff_data, columns=tradeoff_columns)

main_scores_data = []
grouped = tradeoff_df.groupby("Main Criteria", sort=False)

for main_criteria, group in grouped:
    sub_weights = group["Weight"] / 100
    concept_scores = group[concept_cols].multiply(sub_weights, axis=0).sum()
    
    row = {"Criteria": main_criteria, "Weight": main_weights[main_criteria]}
    row.update(concept_scores.to_dict())
    main_scores_data.append(row)

main_scores_df = pd.DataFrame(main_scores_data).set_index("Criteria")

main_weight_pct = main_scores_df["Weight"] / 100
final_total_scores = main_scores_df[concept_cols].multiply(main_weight_pct, axis=0).sum()

total_row = {"Weight": ""}
total_row.update(final_total_scores.to_dict())
main_scores_df.loc["Total score"] = total_row

final_display_df = main_scores_df.rename(columns=concept_names)
actual_concept_names_list = list(concept_names.values())


# ==========================================
# 7. Ultimate Trade-off Ranking
# ==========================================
final_ranking_data = []
for col in concept_cols:
    final_ranking_data.append({
        "Concept": concept_names[col],
        "Final Score": final_total_scores[col]
    })

final_ranking_df = pd.DataFrame(final_ranking_data)
final_ranking_df = final_ranking_df.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
final_ranking_df.index = final_ranking_df.index + 1


# ==========================================
# 8. Visual Sensitivity Analysis (Matplotlib)
# ==========================================

def plot_initial_tradeoff_sensitivity(aerial_df, aquatic_df, valid_combos_df, sweep_pct):
    aerial_base_w = aerial_df.drop("Total score")["Weight"].astype(float)
    aquatic_base_w = aquatic_df.drop("Total score")["Weight"].astype(float)
    
    aerial_scores = aerial_df.drop("Total score")[aerial_vehicle_cols].astype(float)
    aquatic_scores = aquatic_df.drop("Total score")[aquatic_propulsion_cols].astype(float)
    
    base_aq_totals = aquatic_scores.multiply(aquatic_base_w / 100, axis=0).sum()
    base_ae_totals = aerial_scores.multiply(aerial_base_w / 100, axis=0).sum()
    
    combo_names = [f"{row['Vehicle']} & {row['Propulsion']}" for _, row in valid_combos_df.iterrows()]
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle(f"Sensitivity Analysis: Initial Aerial & Aquatic Trade-offs (+/- {sweep_pct}%)", fontsize=18, fontweight='bold')
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(combo_names)))
    color_map = {name: color for name, color in zip(combo_names, colors)}

    for i, target_crit in enumerate(aerial_base_w.index):
        ax = axes[0, i]
        w_base = aerial_base_w[target_crit]
        min_w = max(0.0, w_base - sweep_pct)
        max_w = min(100.0, w_base + sweep_pct)
        weight_range = np.linspace(min_w, max_w, 100)
        
        simulated_scores = {name: [] for name in combo_names}
        
        for w_target in weight_range:
            new_w = aerial_base_w.copy()
            new_w[target_crit] = w_target
            diff = w_target - w_base
            other_sum = 100.0 - w_base
            
            if other_sum > 0:
                for ocrit in new_w.index:
                    if ocrit != target_crit:
                        new_w[ocrit] -= diff * (aerial_base_w[ocrit] / other_sum)
            else:
                for ocrit in new_w.index:
                    if ocrit != target_crit:
                        new_w[ocrit] = (100.0 - w_target) / (len(new_w) - 1)
                        
            new_ae_totals = aerial_scores.multiply(new_w / 100, axis=0).sum()
            
            for _, row in valid_combos_df.iterrows():
                veh, prop = row['Vehicle'], row['Propulsion']
                name = f"{veh} & {prop}"
                score = (new_ae_totals[veh] + base_aq_totals[prop]) / 2
                simulated_scores[name].append(score)
                
        for name in combo_names:
            ax.plot(weight_range, simulated_scores[name], color=color_map[name], linewidth=2)
            
        ax.axvline(x=w_base, color='black', linestyle=':', alpha=0.7)
        ax.set_title(f"Aerial: {target_crit}")
        ax.set_xlabel(f"Weight (%)")
        ax.set_ylabel("Combined Score")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min_w, max_w)

    for i, target_crit in enumerate(aquatic_base_w.index):
        ax = axes[1, i]
        w_base = aquatic_base_w[target_crit]
        min_w = max(0.0, w_base - sweep_pct)
        max_w = min(100.0, w_base + sweep_pct)
        weight_range = np.linspace(min_w, max_w, 100)
        
        simulated_scores = {name: [] for name in combo_names}
        
        for w_target in weight_range:
            new_w = aquatic_base_w.copy()
            new_w[target_crit] = w_target
            diff = w_target - w_base
            other_sum = 100.0 - w_base
            
            if other_sum > 0:
                for ocrit in new_w.index:
                    if ocrit != target_crit:
                        new_w[ocrit] -= diff * (aquatic_base_w[ocrit] / other_sum)
            else:
                for ocrit in new_w.index:
                    if ocrit != target_crit:
                        new_w[ocrit] = (100.0 - w_target) / (len(new_w) - 1)
                        
            new_aq_totals = aquatic_scores.multiply(new_w / 100, axis=0).sum()
            
            for _, row in valid_combos_df.iterrows():
                veh, prop = row['Vehicle'], row['Propulsion']
                name = f"{veh} & {prop}"
                score = (base_ae_totals[veh] + new_aq_totals[prop]) / 2
                simulated_scores[name].append(score)
                
        for name in combo_names:
            ax.plot(weight_range, simulated_scores[name], label=name, color=color_map[name], linewidth=2)
            
        ax.axvline(x=w_base, color='black', linestyle=':', alpha=0.7)
        ax.set_title(f"Aquatic: {target_crit}")
        ax.set_xlabel(f"Weight (%)")
        ax.set_ylabel("Combined Score")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min_w, max_w)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=10, title="Combinations (Top down by baseline score)")
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.83, top=0.90) 


def plot_initial_criteria_removal_bars(aerial_df, aquatic_df, top_combos_df):
    """
    Generates side-by-side grouped bar charts for the initial matrices,
    showing how the Top 6 Combinations react when criteria are removed.
    """
    aerial_base_w = aerial_df.drop("Total score")["Weight"].astype(float)
    aquatic_base_w = aquatic_df.drop("Total score")["Weight"].astype(float)
    
    aerial_scores = aerial_df.drop("Total score")[aerial_vehicle_cols].astype(float)
    aquatic_scores = aquatic_df.drop("Total score")[aquatic_propulsion_cols].astype(float)
    
    base_aq_totals = aquatic_scores.multiply(aquatic_base_w / 100, axis=0).sum()
    base_ae_totals = aerial_scores.multiply(aerial_base_w / 100, axis=0).sum()
    
    combo_names = [f"{row['Vehicle']} & {row['Propulsion']}" for _, row in top_combos_df.iterrows()]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Impact of Eliminating Individual Criteria (Initial Trade-offs)', fontsize=18, fontweight='bold')
    
    num_concepts = len(combo_names)
    width = 0.12
    offsets = np.linspace(-width*(num_concepts-1)/2, width*(num_concepts-1)/2, num_concepts)
    
    # --- Subplot 1: Aerial Removals ---
    ax1 = axes[0]
    scenarios_ae = ["Baseline"] + [f"No {crit}" for crit in aerial_base_w.index]
    scores_ae = {name: [] for name in combo_names}
    
    for _, row in top_combos_df.iterrows():
        veh, prop = row['Vehicle'], row['Propulsion']
        name = f"{veh} & {prop}"
        scores_ae[name].append((base_ae_totals[veh] + base_aq_totals[prop]) / 2)
        
    for crit_to_remove in aerial_base_w.index:
        new_w = aerial_base_w.copy()
        new_w[crit_to_remove] = 0.0
        if new_w.sum() > 0:
            new_w = (new_w / new_w.sum()) * 100.0
        new_totals = aerial_scores.multiply(new_w / 100, axis=0).sum()
        
        for _, row in top_combos_df.iterrows():
            veh, prop = row['Vehicle'], row['Propulsion']
            name = f"{veh} & {prop}"
            scores_ae[name].append((new_totals[veh] + base_aq_totals[prop]) / 2)
            
    x_ae = np.arange(len(scenarios_ae))
    for i, name in enumerate(combo_names):
        ax1.bar(x_ae + offsets[i], scores_ae[name], width, label=name, zorder=3)
        
    ax1.set_ylabel('Combined Score', fontsize=12)
    ax1.set_title('Aerial Criteria Eliminated', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_ae)
    ax1.set_xticklabels(scenarios_ae, fontsize=10, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # --- Subplot 2: Aquatic Removals ---
    ax2 = axes[1]
    scenarios_aq = ["Baseline"] + [f"No {crit}" for crit in aquatic_base_w.index]
    scores_aq = {name: [] for name in combo_names}
    
    for _, row in top_combos_df.iterrows():
        veh, prop = row['Vehicle'], row['Propulsion']
        name = f"{veh} & {prop}"
        scores_aq[name].append((base_ae_totals[veh] + base_aq_totals[prop]) / 2)
        
    for crit_to_remove in aquatic_base_w.index:
        new_w = aquatic_base_w.copy()
        new_w[crit_to_remove] = 0.0
        if new_w.sum() > 0:
            new_w = (new_w / new_w.sum()) * 100.0
        new_totals = aquatic_scores.multiply(new_w / 100, axis=0).sum()
        
        for _, row in top_combos_df.iterrows():
            veh, prop = row['Vehicle'], row['Propulsion']
            name = f"{veh} & {prop}"
            scores_aq[name].append((base_ae_totals[veh] + new_totals[prop]) / 2)
            
    x_aq = np.arange(len(scenarios_aq))
    for i, name in enumerate(combo_names):
        ax2.bar(x_aq + offsets[i], scores_aq[name], width, label=name, zorder=3)
        
    ax2.set_title('Aquatic Criteria Eliminated', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_aq)
    ax2.set_xticklabels(scenarios_aq, fontsize=10, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    # Legend setup
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5), title="Top 6 Concepts")
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, top=0.90)


def plot_weight_variance_graphs(tradeoff_name, eval_matrix, cols, sweep_pct):
    df_base = eval_matrix.drop("Total score")
    baseline_weights = df_base["Weight"].astype(float)
    scores = df_base[cols].astype(float)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{tradeoff_name}: Sensitivity to Weight Variance (+/- {sweep_pct}%)", fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for i, target_crit in enumerate(baseline_weights.index):
        ax = axes[i]
        simulated_scores = {concept: [] for concept in cols}
        
        w_base_target = baseline_weights[target_crit]
        min_w = max(0.0, w_base_target - sweep_pct)
        max_w = min(100.0, w_base_target + sweep_pct)
        weight_range = np.linspace(min_w, max_w, 100)
        
        for w_target in weight_range:
            diff = w_target - w_base_target
            other_sum = 100.0 - w_base_target
            
            new_weights = baseline_weights.copy()
            new_weights[target_crit] = w_target
            
            if other_sum > 0:
                for ocrit in baseline_weights.index:
                    if ocrit != target_crit:
                        new_weights[ocrit] -= diff * (baseline_weights[ocrit] / other_sum)
            else:
                for ocrit in baseline_weights.index:
                    if ocrit != target_crit:
                        new_weights[ocrit] = (100.0 - w_target) / (len(baseline_weights) - 1)
                        
            new_totals = scores.multiply(new_weights / 100, axis=0).sum()
            for concept in cols:
                simulated_scores[concept].append(new_totals[concept])
                
        for concept in cols:
            line_style = '-' if concept == actual_concept_names_list[0] else '--' 
            linewidth = 3 if concept == actual_concept_names_list[0] else 1.5
            
            ax.plot(weight_range, simulated_scores[concept], 
                    label=concept, linestyle=line_style, linewidth=linewidth)
        
        ax.axvline(x=baseline_weights[target_crit], color='black', linestyle=':', label='Baseline Weight')
        ax.set_title(f"Varying Weight of: {target_crit}")
        ax.set_xlabel(f"{target_crit} Weight (%)")
        ax.set_ylabel("Total Score")
        ax.set_xlim(min_w, max_w)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 

def plot_criteria_removal_bars(tradeoff_name, eval_matrix, cols):
    df_base = eval_matrix.drop("Total score")
    baseline_weights = df_base["Weight"].astype(float)
    scores = df_base[cols].astype(float)

    scenarios = ["Baseline"] + [f"No {crit}" for crit in baseline_weights.index]
    concept_scores = {concept: [] for concept in cols}

    baseline_totals = scores.multiply(baseline_weights / 100, axis=0).sum()
    for concept in cols:
        concept_scores[concept].append(baseline_totals[concept])

    for crit_to_remove in baseline_weights.index:
        new_weights = baseline_weights.copy()
        new_weights[crit_to_remove] = 0.0
        
        other_sum = new_weights.sum()
        if other_sum > 0:
            new_weights = (new_weights / other_sum) * 100.0

        new_totals = scores.multiply(new_weights / 100, axis=0).sum()
        for concept in cols:
            concept_scores[concept].append(new_totals[concept])

    x = np.arange(len(scenarios))
    num_concepts = len(cols)
    width = 0.12 
    
    fig, ax = plt.subplots(figsize=(14, 7))
    offsets = np.linspace(-width*(num_concepts-1)/2, width*(num_concepts-1)/2, num_concepts)
    
    for i, concept in enumerate(cols):
        ax.bar(x + offsets[i], concept_scores[concept], width, label=concept, zorder=3)

    ax.set_ylabel('Total Score', fontsize=12)
    ax.set_title(f'{tradeoff_name}: Impact of Eliminating Individual Criteria', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11, fontweight='bold')
    
    ax.axhline(y=baseline_totals.max(), color='black', linestyle='--', alpha=0.5, zorder=2, label='Baseline Winner Score')

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()


# ==========================================
# 9. Final Output Execution
# ==========================================
if __name__ == "__main__":
    pd.options.display.float_format = '{:.3f}'.format

    print("--- AERIAL Concept Evaluation (Dynamically Calculated) ---")
    print(aerial_matrix)
    print("\n" + "="*80 + "\n")

    print("--- AQUATIC Concept Evaluation (Dynamically Calculated) ---")
    print(aquatic_matrix)
    print("\n" + "="*80 + "\n")

    print("--- Top 6 Ranked Compatible Combinations ---")
    print(top_6_results.to_string())
    print("\n" + "="*80 + "\n")

    print("--- Final Trade-off Main Criteria & Total Score ---")
    final_display_df_reset = final_display_df.reset_index() 
    print(final_display_df_reset.to_string(index=False))
    print("\n" + "="*80 + "\n")

    print("--- Ultimate Final Ranking ---")
    print(final_ranking_df.to_string())
    print("\n" + "="*80 + "\n")

    print("Generating Sensitivity Graphs...")
    
    # Plot 1: Initial Aerial & Aquatic Sensitivity (8 graphs, 15 lines each)
    plot_initial_tradeoff_sensitivity(aerial_matrix, aquatic_matrix, results_df, SENSITIVITY_SWEEP_PERCENT)
    
    # Plot 2: Initial Aerial & Aquatic Criteria Elimination Bar Charts (Filtered to Top 6 for readability)
    plot_initial_criteria_removal_bars(aerial_matrix, aquatic_matrix, top_6_results)
    
    # Plot 3: Final Trade-off continuous weight variance
    plot_weight_variance_graphs("Final Concept Trade-off", final_display_df, actual_concept_names_list, SENSITIVITY_SWEEP_PERCENT)
    
    # Plot 4: Final Trade-off criteria elimination
    plot_criteria_removal_bars("Final Concept Trade-off", final_display_df, actual_concept_names_list)
    
    # Display all generated figures
    plt.show()

    #hello