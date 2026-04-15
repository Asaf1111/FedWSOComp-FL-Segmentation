import pandas as pd

# Path to your original server dataset CSV
full_csv_path = "/home/jovyan/FeTSxFedWSO/data_splitting/server/server_benchmark_dataset.csv"

# Path to save the new 100-sample server dataset
small_csv_path = "/home/jovyan/FeTSxFedWSO/data_splitting/server/server_benchmark_dataset_100.csv"

# Load the full server CSV
df = pd.read_csv(full_csv_path)

# Take only the first 100 samples
df_small = df.head(50)

# Save the reduced dataset
df_small.to_csv(small_csv_path, index=False)

print(f"[INFO] Small server dataset created with {len(df_small)} samples at: {small_csv_path}")

