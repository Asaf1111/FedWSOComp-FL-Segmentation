import os
import pandas as pd
import numpy as np

# Root directory
root_dir = "Results/IPTestR"
output_path = os.path.join(root_dir, "IPTest_summary_detailed22.csv")

# Helper to load CSV and exclude 'Average' row
def load_metrics(path):
    df = pd.read_csv(path)
    df = df[df["Sample"] != "Average"]
    return df

# Results container
summary_rows = []

# Traverse each experiment folder
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)

    # Skip non-directories or system/hidden folders
    if not os.path.isdir(folder_path):
        continue
    if folder_name.startswith('.') or 'checkpoints' in folder_name.lower():
        continue

    row = {'Experiment': folder_name}

    # -------- Server --------
    server_csv = os.path.join(folder_path, "server_per_sample.csv")
    if os.path.isfile(server_csv):
        df = load_metrics(server_csv)
        row.update({
            'Server Dice Mean': round(np.nanmean(df['Dice'].values), 4),
            'Server Dice Std': round(np.nanstd(df['Dice'].values), 4),
            'Server HD95 Mean': round(np.nanmean(df['HD95'].values), 4),
            'Server HD95 Std': round(np.nanstd(df['HD95'].values), 4),
        })
    else:
        row.update({
            'Server Dice Mean': 0, 'Server Dice Std': 0,
            'Server HD95 Mean': 0, 'Server HD95 Std': 0,
        })

    # -------- Clients 1 to 4 --------
    for cid in range(1, 5):
        client_csv = os.path.join(folder_path, f"client{cid}_per_sample.csv")
        if os.path.isfile(client_csv):
            df = load_metrics(client_csv)
            row.update({
                f'Client{cid} Dice Mean': round(np.nanmean(df['Dice'].values), 4),
                f'Client{cid} Dice Std': round(np.nanstd(df['Dice'].values), 4),
                f'Client{cid} HD95 Mean': round(np.nanmean(df['HD95'].values), 4),
                f'Client{cid} HD95 Std': round(np.nanstd(df['HD95'].values), 4),
            })
        else:
            row.update({
                f'Client{cid} Dice Mean': 0,
                f'Client{cid} Dice Std': 0,
                f'Client{cid} HD95 Mean': 0,
                f'Client{cid} HD95 Std': 0,
            })

    summary_rows.append(row)

# Save result to CSV
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(output_path, index=False)
print(f" Summary saved to: {output_path}")
