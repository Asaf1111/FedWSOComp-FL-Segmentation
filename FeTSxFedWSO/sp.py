import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# Paths
# ==============================
original_csv_dir = "Results/results_csvs"
ablation_csv_dir = "Revision2/RCSV"
plot_dir = "final_multi_panel_overlay_clean"
os.makedirs(plot_dir, exist_ok=True)

ALLOWED_C = {3, 32, 64}
ALLOWED_S = {20, 30, 60}

# ==============================
# COLORS (baseline unchanged)
# ==============================
BASE_COLOR = "#e69f00"   # ORIGINAL orange (FedWSOComp)
BAND_COLOR = "#ff69b4"   # ORIGINAL pink band (±1 std)

# Clear, distinct ablation colors
CH_COLOR = {
    3:  "#0057B8",   # deep blue
    32: "#6A0DAD",   # purple
    64: "#B22222",   # dark red
}
SH_COLOR = {
    20: "#006400",   # dark green
    30: "#008080",   # teal
    60: "#B8860B",   # gold
}

LW_BASE = 2.6
LW_ABL = 3.0

# ==============================
# Helpers
# ==============================
def get_base(filename: str) -> str:
    return re.sub(r"R\d+", "", filename.replace(".csv", ""))


def extract_mode(name: str) -> str:
    lower = name.lower()
    return "non-IID" if ("non" in lower and "iid" in lower) else "IID"


def parse_full_CS(name: str):
    lower = name.lower()
    c = re.search(r"c(\d+)", lower)
    s = re.search(r"sr?(\d+)", lower)
    if c and s:
        return int(c.group(1)), int(s.group(1))
    return None, None


def parse_ablation(name: str):
    lower = name.lower()
    if lower.startswith("ch"):
        m = re.search(r"ch(\d+)", lower)
        if m:
            return "CH", int(m.group(1)), 0
    if lower.startswith("sh"):
        m = re.search(r"sh(\d+)", lower)
        if m:
            return "SH", 0, int(m.group(1))
    return None, None, None


def mean_std(dfs, y_col="dice_wt", x_col="round"):
    rounds = dfs[0][x_col].values
    y = np.stack([d[y_col].values for d in dfs], axis=0)
    return rounds, y.mean(0), y.std(0)


def load_grouped(csv_dir: str):
    groups = {}
    for f in os.listdir(csv_dir):
        if f.endswith(".csv"):
            base = get_base(f)
            df = pd.read_csv(os.path.join(csv_dir, f))
            df.columns = df.columns.str.lower()
            groups.setdefault(base, []).append(df)
    return groups


# ==============================
# Load ORIGINAL
# ==============================
orig = load_grouped(original_csv_dir)
full_pairs = {}

for base, dfs in orig.items():
    c, s = parse_full_CS(base)
    if c is None or s is None:
        continue
    if c not in ALLOWED_C or s not in ALLOWED_S:
        continue

    mode = extract_mode(base)
    key = f"C{c}_S{s}"

    full_pairs.setdefault(key, {
        "C": c, "S": s,
        "title": f"C{c} S{s}",
        "IID": None, "non-IID": None
    })
    full_pairs[key][mode] = dfs


# ==============================
# Load ABLATIONS
# ==============================
abl = load_grouped(ablation_csv_dir)
ablation = {}

for base, dfs in abl.items():
    t, c, s = parse_ablation(base)
    if t == "CH" and c in ALLOWED_C:
        k = f"CH{c}"
    elif t == "SH" and s in ALLOWED_S:
        k = f"SH{s}"
    else:
        continue

    mode = extract_mode(base)
    ablation.setdefault(k, {"IID": None, "non-IID": None})
    ablation[k][mode] = dfs


# ==============================
# Build plot list
# ==============================
pairs = []
for _, p in full_pairs.items():
    c, s = p["C"], p["S"]
    has_ch = f"CH{c}" in ablation
    has_sh = f"SH{s}" in ablation
    if not (has_ch or has_sh):
        continue

    p2 = dict(p)
    p2["CH_key"] = f"CH{c}" if has_ch else None
    p2["SH_key"] = f"SH{s}" if has_sh else None
    pairs.append(p2)

pairs.sort(key=lambda x: (x["C"], x["S"]))


# ==============================
# Plot (original layout + correct legend)
# ==============================
def plot(pairs, out):
    n = len(pairs)
    if n == 0:
        print("[WARN] Nothing to plot.")
        return

    cols = 4
    rows = (n // 2) + (1 if (n % 2) else 0)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 4.8, rows * 4.6),
        sharex=False,
        sharey=True
    )
    axes = np.array(axes).reshape(rows, cols)

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 16,
    })

    used = set()

    def pair_position(i):
        # center last pair if odd
        if (n % 2 == 1) and (i == n - 1):
            return rows - 1, 1
        return i // 2, (i % 2) * 2

    for i, p in enumerate(pairs):
        r, c0 = pair_position(i)
        ax_iid = axes[r, c0]
        ax_non = axes[r, c0 + 1]
        used |= {(r, c0), (r, c0 + 1)}

        ax_iid.text(
            -0.18, 1.12, f"{chr(97+i)})",
            transform=ax_iid.transAxes,
            fontsize=16,
            fontweight="bold"
        )

        for ax, mode in [(ax_iid, "IID"), (ax_non, "non-IID")]:
            if p[mode] is None:
                ax.set_visible(False)
                continue

            # Baseline full
            r0, m0, s0 = mean_std(p[mode])
            ax.plot(r0, m0, lw=LW_BASE, color=BASE_COLOR, label="FedWSOComp")
            ax.fill_between(r0, m0 - s0, m0 + s0, color=BAND_COLOR, alpha=0.30, label="±1 std")

            # CH ablation
            if p["CH_key"] and ablation[p["CH_key"]][mode] is not None:
                r1, m1, _ = mean_std(ablation[p["CH_key"]][mode])
                ax.plot(r1, m1, lw=LW_ABL, color=CH_COLOR[p["C"]], label=f"CH{p['C']}")

            # SH ablation
            if p["SH_key"] and ablation[p["SH_key"]][mode] is not None:
                r2, m2, _ = mean_std(ablation[p["SH_key"]][mode])
                ax.plot(r2, m2, lw=LW_ABL, color=SH_COLOR[p["S"]], label=f"SH{p['S']}")

            ax.set_title(f"{p['title']} {mode}", fontweight="bold")
            ax.set_xticks(np.arange(0, 26, 5))
            ax.set_ylim(0, 1.05)
            ax.grid(True, ls="--", alpha=0.4)
            ax.tick_params(labelbottom=True)

    # Hide unused axes
    for rr in range(rows):
        for cc in range(cols):
            if (rr, cc) not in used:
                axes[rr, cc].set_visible(False)

    # Layout (reserve space on right for legend)
    plt.tight_layout(rect=[0.10, 0.18, 0.88, 0.96])
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.18, top=0.96)

    fig.supylabel("Dice Score", x=0.04, y=0.55, fontsize=18, fontweight="bold")
    fig.supxlabel("Communication Rounds", y=0.12, fontsize=18, fontweight="bold")

    # ---- Correct legend: gather from ALL visible axes and deduplicate ----
    all_h, all_l = [], []
    for ax in axes.flatten():
        if ax.get_visible():
            h, l = ax.get_legend_handles_labels()
            all_h.extend(h)
            all_l.extend(l)

    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(all_h, all_l):
        if l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)

    # Anchor legend near last pair
    lr, lc0 = pair_position(n - 1)
    anchor_ax = axes[lr, lc0 + 1]
    if not anchor_ax.get_visible():
        anchor_ax = axes[lr, lc0]
    pos = anchor_ax.get_position()

    fig.legend(
        uniq_h, uniq_l,
        loc="center left",
        bbox_to_anchor=(pos.x1 + 0.02, (pos.y0 + pos.y1) / 2),
        frameon=False
    )

    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure: {out}")


# ==============================
# Run
# ==============================
out_path = os.path.join(plot_dir, "FedWSOComp_with_ablation_original_colors_FIXED_LEGEND.pdf")
plot(pairs, out_path)
