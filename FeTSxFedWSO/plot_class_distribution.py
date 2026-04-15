import numpy as np
import matplotlib.pyplot as plt

clients      = ["Client 1", "Client 2", "Client 3", "Client 4"]
iid_counts   = [100, 100, 100, 100]
non_iid_cnts = [511, 382, 151, 140]

x = np.arange(len(clients))
width = 0.37

fig, ax = plt.subplots(figsize=(12, 7))

rects_iid = ax.bar(x - width/2, iid_counts, width, label="IID", color="#1f77b4")
rects_non = ax.bar(x + width/2, non_iid_cnts, width, label="Non-IID", color="#ff7f0e")

# Axis labels (scaled down)
ax.set_ylabel("Sample Count", fontsize=18, fontweight="bold")
ax.set_xlabel("Client", fontsize=18, fontweight="bold")

# ✅ Remove title
# ax.set_title("Samples per Client: IID vs Non-IID", fontsize=24, fontweight="bold")

# X-tick labels NOT bold (kept readable)
ax.set_xticks(x)
ax.set_xticklabels(clients, fontsize=18, fontweight="normal")

# Y-ticks
ax.tick_params(axis="y", labelsize=16)

# Increase vertical space
max_val = max(max(iid_counts), max(non_iid_cnts))
ax.set_ylim(0, max_val + 70)

# Legend
ax.legend(loc="upper right", fontsize=16, frameon=False)

# Bold labels above bars
def autolabel(rects):
    for r in rects:
        h = r.get_height()
        ax.annotate(
            f"{int(h)}",
            xy=(r.get_x() + r.get_width() / 2, h),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold"
        )

autolabel(rects_iid)
autolabel(rects_non)

fig.tight_layout()
plt.savefig("samples_per_client_IID_nonIID_clean.pdf", dpi=400, bbox_inches="tight")
plt.show()
