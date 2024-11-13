import os
import pandas as pd
import matplotlib.pyplot as plt

# Define directory paths for the workers
base_dir = "train_data"  # Update this with your base directory path
worker_counts = ["_1", "_4", "_8"]
file_limit = 5
smoothing_window = 10  # Adjust this to control the degree of smoothing

# Initialize data containers
worker_data = {}

# Read and process the CSV files for each worker count
for count in worker_counts:
    files = []
    for folder in ["w_1", "w_4", "w_8"]:
        folder_path = os.path.join(base_dir, folder)
        # Collect only the first five files matching the worker count (_1, _4, _8)
        matched_files = sorted([f for f in os.listdir(folder_path) if f.endswith(f"workers{count}.csv")])[:file_limit]
        files.extend([os.path.join(folder_path, file) for file in matched_files])

    # Check column names by reading the first file
    if files:
        sample_df = pd.read_csv(files[0])
        print("Column names in the file:", sample_df.columns)
        # Update column names here if they differ from "Training Steps" or "Validation Episode Reward Average"

    # Read data, assuming "Training Steps" and "Validation Episode Reward Average" columns exist
    data_frames = [pd.read_csv(file) for file in files]
    combined_data = pd.concat(data_frames).groupby(level=0).mean()

    # Apply rolling average for smoothing
    combined_data['Training Steps'] = combined_data['Training Steps'].rolling(window=smoothing_window).mean()
    combined_data['Validation Episode Reward Average'] = combined_data['Validation Episode Reward Average'].rolling(
        window=smoothing_window).mean()

    worker_data[count] = combined_data

# Plotting
plt.figure(figsize=(10, 6))
styles = ['solid', 'dashed', 'dotted']
colors = ['purple', 'orange', 'green']
labels = ['1 Worker', '4 Workers', '8 Workers']

# Plot each worker count's smoothed data
for i, (count, data) in enumerate(worker_data.items()):
    plt.plot(data['Training Steps'], data['Validation Episode Reward Average'], label=labels[i], linestyle=styles[i],
             color=colors[i])

# Customize plot
plt.xlabel('Total Training Steps')
plt.ylabel('Reward')
plt.title('Training Steps vs Reward for Different Worker Counts (Smoothed)')
plt.legend(loc='upper left', frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()
