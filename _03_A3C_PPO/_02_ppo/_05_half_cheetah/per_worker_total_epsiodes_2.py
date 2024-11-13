import os
import pandas as pd
import matplotlib.pyplot as plt

# Define directory paths and settings
base_dir = "train_data"  # Update this with your base directory path
worker_folders = ["w_1", "w_4", "w_8"]  # Different worker amounts
run_indices = ["_2", "_3"]  # Different run indexes
file_limit = 5

# Initialize data containers
worker_data = {}

# Process files by worker amount and run index
for folder in worker_folders:
    files = []
    folder_path = os.path.join(base_dir, folder)
    # Collect first five files for each worker count with specified run indexes
    matched_files = sorted([
        f for f in os.listdir(folder_path)
        if any(f.endswith(f"{run}.csv") for run in run_indices)
    ])[:file_limit]
    files.extend([os.path.join(folder_path, file) for file in matched_files])

    # Read and clean column names for each data frame
    data_frames = []
    for file in files:
        df = pd.read_csv(file)

        # Strip any whitespace from column names
        df.columns = df.columns.str.strip()

        # Convert "Episode" and "Episode Reward" columns to numeric after cleaning
        # df['Episode'] = pd.to_numeric(df['Episode'], errors='coerce')
        # df['Episode Reward'] = pd.to_numeric(df['Episode Reward'], errors='coerce')

        data_frames.append(df)

    # Combine data and calculate mean only on numeric columns
    combined_data = pd.concat(data_frames).groupby(level=0).mean(numeric_only=True)

    # Store averaged data for each worker amount
    worker_data[folder] = combined_data

# Plotting separate graphs for each worker amount
for i, (folder, data) in enumerate(worker_data.items()):
    plt.figure(figsize=(8, 6))
    plt.plot(data['Episode'], data['Episode Reward'], label=f'{folder} Workers', color=['blue', 'green', 'orange'][i])

    # Customize each plot
    plt.xlabel('Total Episode')
    plt.ylabel('Reward')
    plt.title(f'Episode vs Reward for {folder.replace("w_", "")} Workers')
    plt.legend(loc='upper left', frameon=False)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
