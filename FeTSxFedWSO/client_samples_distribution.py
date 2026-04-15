import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file paths
iid_paths = [f"data_splitting/clients/iid/client{i}_iid_dataset.csv" for i in range(1, 5)]
noniid_paths = [f"data_splitting/clients/non_iid/client{i}_noniid_dataset.csv" for i in range(1, 5)]

iid_counts = []
noniid_counts = []

# Load sample counts
for iid, noniid in zip(iid_paths, noniid_paths):
    try:
        iid_counts.append(pd.read_csv(iid).shape[0])
    except Exception as e:
        print(f"Could not read {iid}: {e}")
        iid_counts.append(0)
    try:
        noniid_counts.append(pd.read_csv(noniid).shape[0])
    except Exception as e:
        print(f"Could not read {noniid}: {e}")
        noniid_counts.append(0)

# Plot setup
clients = [f"Client {i}" for i in range(1, 5)]
x = range(len(clients))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x, iid_counts, width=bar_width, label="IID")
bars2 = ax.bar([i + bar_width for i in x], noniid_counts, width=bar_width, label="Non-IID")

# Add labels and legend
ax.set_xticks([i + bar_width / 2 for i in x])
ax.set_xticklabels(clients)
ax.set_ylabel("Sample Count")
ax.set_title("Samples per Client: IID vs Non-IID")
ax.legend()

# Annotate bar values
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# Save plot
os.makedirs("client_sample_distribution", exist_ok=True)
plt.savefig("client_sample_distribution/client_sample_distribution.pdf", dpi=300)
plt.show()
