import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# Paths
# ==============================
csv_dir = "Results/results_csvs"
plot_dir = "final_multi_panel_centered_lastpair_clean_v6"
os.makedirs(plot_dir, exist_ok=True)


# ==============================
# Helpers
# ==============================
def get_experiment_base(filename: str) -> str:
    """Remove run markers like R1, R2 etc. and strip .csv."""
    return re.sub(r"R\d+", "", filename.replace(".csv", ""))


def is_fedavg_experiment(name: str) -> bool:
    """Identify FedAvg experiments to exclude from compressed figure."""
    return "fedavg" in name.lower()


def extract_c_s_iid(exp_name: str):
    """
    Parse C, S, and IID/non-IID from the experiment base name.

    Returns:
        pair_key   e.g. "C3_S60"
        pair_title e.g. "C3 S60"
        iid_str    "IID" / "non-IID" / None
    """
    lower = exp_name.lower()

    # Cluster C..
    c_match = re.search(r"c(\d+)", lower)
    c_val = c_match.group(1) if c_match else None

    # Sparsity S.. or SR..
    s_val = None
    sr_match = re.search(r"sr(\d+)", lower)
    if sr_match:
        s_val = sr_match.group(1)
    else:
        s_match = re.search(r"s(\d+)", lower)
        if s_match:
            s_val = s_match.group(1)

    # IID / non-IID
    iid_str = None
    if re.search(r"non[_\-]?iid", lower):
        iid_str = "non-IID"
    elif "iid" in lower:
        iid_str = "IID"

    if c_val is not None and s_val is not None:
        pair_key = f"C{c_val}_S{s_val}"
        pair_title = f"C{c_val} S{s_val}"
    else:
        pair_key = exp_name
        pair_title = exp_name.replace("_", " ").strip()

    return pair_key, pair_title, iid_str


def compute_rows_cols_for_pairs(num_pairs: int, pairs_per_row: int = 2, cols: int = 4):
    """Each pair uses 2 columns. 4 columns total → 2 pairs per full row."""
    full_rows = num_pairs // pairs_per_row
    remaining_pairs = num_pairs % pairs_per_row
    rows = full_rows + (1 if remaining_pairs else 0)
    return rows, cols, full_rows, remaining_pairs


# ==============================
# Plotting
# ==============================
def plot_multi_panel_pairs(pair_list, out_path: str):
    num_pairs = len(pair_list)
    pairs_per_row = 2
    cols = 4

    rows, cols, full_rows, remaining_pairs = compute_rows_cols_for_pairs(
        num_pairs, pairs_per_row=pairs_per_row, cols=cols
    )

    # sharex=False so each subplot keeps its own x ticks
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 4.5, rows * 4.3),
        sharex=False,
        sharey=True,
    )
    axes = np.array(axes).reshape(rows, cols)

    used_axes = set()
    global_handles, global_labels = None, None

    # Fonts
    plt.rcParams.update(
        {
            "axes.titlesize": 17,
            "axes.labelsize": 19,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 17,
        }
    )

    def pair_position(pair_idx: int):
        """Return (row_idx, col_start) for this pair (2 subplots)."""
        if pair_idx < full_rows * pairs_per_row:
            row_idx = pair_idx // pairs_per_row
            col_start = (pair_idx % pairs_per_row) * 2
        else:
            # Last row: center pair in columns 1 and 2
            row_idx = rows - 1
            col_start = 1
        return row_idx, col_start

    # ---- Draw all pairs ----
    for pair_idx, pair in enumerate(pair_list):
        row_idx, col_start = pair_position(pair_idx)
        ax_iid = axes[row_idx, col_start]
        ax_non = axes[row_idx, col_start + 1]
        used_axes.add((row_idx, col_start))
        used_axes.add((row_idx, col_start + 1))

        # Panel label a), b), ...
        letter = chr(ord("a") + pair_idx)
        pair_label = f"{letter})"

        # ----- IID subplot -----
        if pair.get("IID") is not None:
            exp_name, dfs = pair["IID"]
            rounds = dfs[0]["round"].values
            all_dice = np.stack([df["dice_wt"].values for df in dfs], axis=0)

            mean_dice = np.mean(all_dice, axis=0)
            std_dice = np.std(all_dice, axis=0)

            ax_iid.plot(
                rounds,
                mean_dice,
                linewidth=2.3,
                color="#e69f00",
                label="Mean Dice",
            )
            ax_iid.fill_between(
                rounds,
                mean_dice - std_dice,
                mean_dice + std_dice,
                color="#ff69b4",
                alpha=0.32,
                label="±1 std",
            )

            if global_handles is None:
                global_handles, global_labels = ax_iid.get_legend_handles_labels()

            ax_iid.set_title(
                f"{pair['pair_title']} IID", fontsize=11, fontweight="bold"
            )
            ax_iid.set_xticks(np.arange(0, 26, 5))
            ax_iid.set_yticks(np.round(np.arange(0.0, 1.01, 0.2), 2))
            ax_iid.set_ylim(0.0, 1.05)
            ax_iid.grid(True, linestyle="--", alpha=0.5)
            ax_iid.tick_params(axis="both", which="major", labelsize=10)

            ax_iid.text(
                -0.15,
                1.10,
                pair_label,
                transform=ax_iid.transAxes,
                fontsize=13,
                fontweight="bold",
                va="top",
                ha="left",
            )
        else:
            ax_iid.set_visible(False)

        # ----- non-IID subplot -----
        if pair.get("non-IID") is not None:
            exp_name, dfs = pair["non-IID"]
            rounds = dfs[0]["round"].values
            all_dice = np.stack([df["dice_wt"].values for df in dfs], axis=0)

            mean_dice = np.mean(all_dice, axis=0)
            std_dice = np.std(all_dice, axis=0)

            ax_non.plot(
                rounds,
                mean_dice,
                linewidth=2.3,
                color="#e69f00",
                label="Mean Dice",
            )
            ax_non.fill_between(
                rounds,
                mean_dice - std_dice,
                mean_dice + std_dice,
                color="#ff69b4",
                alpha=0.32,
                label="±1 std",
            )

            if global_handles is None:
                global_handles, global_labels = ax_non.get_legend_handles_labels()

            ax_non.set_title(
                f"{pair['pair_title']} non-IID", fontsize=11, fontweight="bold"
            )
            ax_non.set_xticks(np.arange(0, 26, 5))
            ax_non.set_yticks(np.round(np.arange(0.0, 1.01, 0.2), 2))
            ax_non.set_ylim(0.0, 1.05)
            ax_non.grid(True, linestyle="--", alpha=0.5)
            ax_non.tick_params(axis="both", which="major", labelsize=10)
        else:
            ax_non.set_visible(False)

    # Hide unused axes (only in last row)
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in used_axes:
                axes[r, c].set_visible(False)

    # Force x tick labels visible on all used subplots
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if ax.get_visible():
                ax.tick_params(axis="x", labelbottom=True)

    # Global Y label (centered vertically)
    fig.supylabel(
        "Dice Score",
        fontsize=14,
        fontweight="bold",
        x=0.03,
        y=0.60,   # center in figure coordinates
    )

    # Layout: create space on the right for the legend near i)
    plt.tight_layout(rect=[0.06, 0.20, 0.90, 0.96])  # right=0.90 → free space on right
    fig.subplots_adjust(left=0.09, right=0.90, top=0.96, bottom=0.20)

    # Global X label (under plots)
    fig.supxlabel(
        "Communication Rounds",
        fontsize=14,
        fontweight="bold",
        y=0.14,
    )

    # ---- Legend near i) plot on the RIGHT ----
    if global_handles is not None:
        last_idx = num_pairs - 1
        last_row, last_col_start = pair_position(last_idx)
        # non-IID of last pair (panel i)
        ax_right = axes[last_row, last_col_start + 1]
        pos = ax_right.get_position()
        x_leg = pos.x1 + 0.02              # a bit to the right of i)
        y_leg = (pos.y0 + pos.y1) / 2.0    # vertically centered w.r.t. i)

        fig.legend(
            global_handles,
            global_labels,
            loc="center left",
            bbox_to_anchor=(x_leg, y_leg),
            ncol=1,
            frameon=False,
        )

    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure: {out_path}")


# ==============================
# Load & group experiments
# ==============================
experiment_groups: dict[str, list[pd.DataFrame]] = {}

for file in os.listdir(csv_dir):
    if not file.endswith(".csv"):
        continue

    base = get_experiment_base(file)
    if is_fedavg_experiment(base):
        continue  # Skip FedAvg

    df = pd.read_csv(os.path.join(csv_dir, file))
    df = df.rename(columns=lambda x: x.strip().lower())
    experiment_groups.setdefault(base, []).append(df)

# Build IID/non-IID pairs per (C, S)
pairs = {}
for exp_name, dfs in experiment_groups.items():
    pair_key, pair_title, iid_str = extract_c_s_iid(exp_name)

    if pair_key not in pairs:
        pairs[pair_key] = {
            "pair_key": pair_key,
            "pair_title": pair_title,
            "IID": None,
            "non-IID": None,
        }

    if iid_str == "IID":
        pairs[pair_key]["IID"] = (exp_name, dfs)
    elif iid_str == "non-IID":
        pairs[pair_key]["non-IID"] = (exp_name, dfs)
    else:
        if pairs[pair_key]["IID"] is None:
            pairs[pair_key]["IID"] = (exp_name, dfs)
        else:
            pairs[pair_key]["non-IID"] = (exp_name, dfs)

pair_list = [pairs[k] for k in sorted(pairs.keys())]

out_path = os.path.join(
    plot_dir, "FedWSOcomp_pairs_IID_nonIID_centered_lastpair_clean_v6.pdf"
)
plot_multi_panel_pairs(pair_list, out_path)
