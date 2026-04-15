import os

# Define your CSV results directory path
csv_dir = "Results/results_csvs" 

# Retrieve and print all CSV filenames in the directory
csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

print("List of CSV result files:")
for filename in csv_files:
    print(filename)

import os
import pandas as pd
import numpy as np
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
# Directory containing your CSV files
csv_dir = "Results/results_csvs" 

# Helper function to group filenames by experiment (ignoring repetitions)
def get_experiment_group(filename):
    return re.sub(r'R\d+', '', filename.replace('.csv', ''))

experiment_groups = {}

# Process each CSV file and group results
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        group_name = get_experiment_group(filename)
        file_path = os.path.join(csv_dir, filename)
        df = pd.read_csv(file_path)

        final_dice_score = df.loc[df['round'] == 25, 'dice_wt'].values
        if final_dice_score.size == 0:
            print(f"Round 25 not found in {filename}, skipping.")
            continue
        final_dice_score = final_dice_score[0]

        if group_name not in experiment_groups:
            experiment_groups[group_name] = []
        experiment_groups[group_name].append(final_dice_score)

# Create summary with mean ± std format
results_summary = []
for experiment, scores in experiment_groups.items():
    mean_dice = np.mean(scores)
    std_dice = np.std(scores)
    results_summary.append({
        "Experiment": experiment,
        "Dice Score (mean ± std)": f"{mean_dice:.4f} ± {std_dice:.4f}",
        "Runs": len(scores)
    })

# Save to CSV
results_summary_df = pd.DataFrame(results_summary)
output_file = "final_results_summary.csv"
results_summary_df.to_csv(output_file, index=False)

print(f"Results saved to '{output_file}'")
def get_experiment_base(filename):
    return re.sub(r'R\d+', '', filename.replace('.csv', ''))

# Collect CSVs into structured groups
experiment_groups = {}

for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        experiment_base = get_experiment_base(filename)
        file_path = os.path.join(csv_dir, filename)
        df = pd.read_csv(file_path)

        if experiment_base not in experiment_groups:
            experiment_groups[experiment_base] = []
        
        repetition_label = re.findall(r'(R\d+)', filename)
        rep = repetition_label[0] if repetition_label else 'R1'

        df['run'] = rep
        experiment_groups[experiment_base].append(df)

# Plot settings
sns.set(style="whitegrid")
plot_dir = "experiment_plots"
os.makedirs(plot_dir, exist_ok=True)

# Plot each experiment individually
for experiment, dfs in experiment_groups.items():
    plt.figure(figsize=(10, 6))
    
    for df in dfs:
        run_label = df['run'].iloc[0]
        plt.plot(df['round'], df['dice_wt'], marker='o', label=f'Run {run_label}')

    plt.title(f"Dice Score vs. Rounds: {experiment}")
    plt.xlabel("Round")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"{experiment}_dice_plot.pdf")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved: {plot_path}")
# Re-importing necessary modules after kernel reset




import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import math

# Directory paths
csv_dir = "Results/results_csvs"  # Correct folder path
plot_dir = "compressed_5_figures_no_fedavg"
os.makedirs(plot_dir, exist_ok=True)

# Normalize experiment name
def get_experiment_base(filename):
    return re.sub(r'R\d+', '', filename.replace('.csv', ''))

# Filter out FedAvg experiments
def is_fedavg_experiment(name):
    return 'fedavg' in name.lower()

# Load and group CSVs
experiment_groups = {}
for file in os.listdir(csv_dir):
    if file.endswith(".csv"):
        base = get_experiment_base(file)
        if is_fedavg_experiment(base):
            continue  # skip FedAvg experiments
        df = pd.read_csv(os.path.join(csv_dir, file))
        df = df.rename(columns=lambda x: x.strip().lower())
        experiment_groups.setdefault(base, []).append(df)

# Prepare grouped plotting (fixed to 5 figures)
experiment_items = list(experiment_groups.items())
total_exps = len(experiment_items)
target_figures = 5
plots_per_fig = math.ceil(total_exps / target_figures)

def compute_subplot_grid(n):
    cols = math.ceil(np.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

# Plot experiments across 5 figures
for fig_idx in range(target_figures):
    start = fig_idx * plots_per_fig
    end = min(start + plots_per_fig, total_exps)
    chunk = experiment_items[start:end]

    rows, cols = compute_subplot_grid(len(chunk))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = np.array(axes).reshape(-1)

    for ax, (exp_name, dfs) in zip(axes, chunk):
        rounds = dfs[0]['round']
        all_dice = np.stack([df['dice_wt'].values for df in dfs], axis=0)
        mean_dice = np.mean(all_dice, axis=0)
        std_dice = np.std(all_dice, axis=0)

        # Plot mean and std
        ax.plot(rounds, mean_dice, color='#e69f00', linewidth=2.5, label='Mean')
        ax.fill_between(rounds, mean_dice - std_dice, mean_dice + std_dice,
                        color='#ff69b4', alpha=0.35, label='±1 std')

        clean_title = exp_name.replace("_", " ").replace("FedWSOComp", "FedWSOcomp")
        ax.set_title(clean_title, fontsize=9)
        ax.set_xticks(np.arange(0, 26, 5))
        ax.set_yticks(np.round(np.arange(0.0, 1.01, 0.2), 2))
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.label_outer()

    for ax in axes[len(chunk):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(os.path.join(plot_dir, f"grouped_experiments_fig_{fig_idx + 1}.pdf"), dpi=300)
    plt.close()




import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import math

# Directory paths
csv_dir = "Results/results_csvs"  # Correct folder path
plot_dir = "compressed_5_figures_no_fedavg"
os.makedirs(plot_dir, exist_ok=True)

# Normalize experiment name
def get_experiment_base(filename):
    return re.sub(r'R\d+', '', filename.replace('.csv', ''))

# Filter out FedAvg experiments
def is_fedavg_experiment(name):
    return 'fedavg' in name.lower()

# Load and group CSVs
experiment_groups = {}
for file in os.listdir(csv_dir):
    if file.endswith(".csv"):
        base = get_experiment_base(file)
        if is_fedavg_experiment(base):
            continue  # skip FedAvg experiments
        df = pd.read_csv(os.path.join(csv_dir, file))
        df = df.rename(columns=lambda x: x.strip().lower())
        experiment_groups.setdefault(base, []).append(df)

# Prepare grouped plotting (fixed to 5 figures)
experiment_items = list(experiment_groups.items())
total_exps = len(experiment_items)
target_figures = 5
plots_per_fig = math.ceil(total_exps / target_figures)

def compute_subplot_grid(n):
    cols = math.ceil(np.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

# Plot experiments across 5 figures
for fig_idx in range(target_figures):
    start = fig_idx * plots_per_fig
    end = min(start + plots_per_fig, total_exps)
    chunk = experiment_items[start:end]

    rows, cols = compute_subplot_grid(len(chunk))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = np.array(axes).reshape(-1)

    for ax, (exp_name, dfs) in zip(axes, chunk):
        rounds = dfs[0]['round']
        all_dice = np.stack([df['dice_wt'].values for df in dfs], axis=0)
        mean_dice = np.mean(all_dice, axis=0)
        std_dice = np.std(all_dice, axis=0)

        # Plot mean and std
        ax.plot(rounds, mean_dice, color='#e69f00', linewidth=2.5, label='Mean')
        ax.fill_between(rounds, mean_dice - std_dice, mean_dice + std_dice,
                        color='#ff69b4', alpha=0.35, label='±1 std')

        clean_title = exp_name.replace("_", " ").replace("FedWSOComp", "FedWSOcomp")
        ax.set_title(clean_title, fontsize=9)
        ax.set_xticks(np.arange(0, 26, 5))
        ax.set_yticks(np.round(np.arange(0.0, 1.01, 0.2), 2))
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.label_outer()

    for ax in axes[len(chunk):]:
        ax.set_visible(False)

    # Collect handles/labels from all subplots
    handles, labels = [], []
    for ax in axes[:len(chunk)]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicate labels
    by_label = dict(zip(labels, handles))

    # Add a single legend below the figure
    fig.legend(by_label.values(), by_label.keys(),
               loc='upper center',
               bbox_to_anchor=(0.5, -0.0007),  # push legend below
               ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0.02, 1, 1])  # leave bottom space
    fig.savefig(os.path.join(plot_dir, f"grouped_experiments_fig_{fig_idx + 1}.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)







import os
import pandas as pd

# Directory containing your result CSVs
csv_dir = "Results/results_csvs"  # Update as needed
output_csv = "summary_final_dice_hd95.csv"

# Collect summary results
summary_data = []

for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        path = os.path.join(input_dir, file)
        df = pd.read_csv(path)

        # Extract final row (average)
        last_row = df.iloc[-1]
        dice = last_row['Dice']
        hd95 = last_row['HD95']
        experiment_name = file.replace(".csv", "")

        summary_data.append({
            "Experiment": experiment_name,
            "Final Dice": round(dice, 4),
            "Final HD95": round(hd95, 4)
        })

# Create DataFrame and save
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(output_csv, index=False)

print("✅ Summary saved to:", output_csv)
print(summary_df)