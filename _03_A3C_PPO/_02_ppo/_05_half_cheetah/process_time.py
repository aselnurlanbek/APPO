import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define directory paths for the workers
base_dir = "train_data"  # Update this with your base directory path
workers = ["w_1", "w_4", "w_8"]
file_limit = 5

# Initialize data containers
worker_data = {}


# Function to convert time string (e.g., "00:00:11") to seconds
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


# Read and process the CSV files
for worker in workers:
    worker_dir = os.path.join(base_dir, worker)
    files = sorted([f for f in os.listdir(worker_dir) if f.startswith("master_processor") and f.endswith(".csv")])[
            :file_limit]
    data_frames = []
    for file in files:
        df = pd.read_csv(os.path.join(worker_dir, file))
        df['Total Process Time'] = df['Total Process Time'].apply(time_to_seconds)
        data_frames.append(df)
    combined_data = pd.concat(data_frames).groupby(level=0).mean()
    worker_data[worker] = combined_data

# Style configurations
plt.figure(figsize=(15, 5))
styles = ['solid', 'dashed', 'dotted']  # Line styles for each worker
colors = ['blue', 'green', 'red']  # Colors for each worker
labels = ['Worker 1', 'Worker 4', 'Worker 8']

# Plot each worker's data with shaded error regions
for i, (worker, data) in enumerate(worker_data.items()):
    # Sample error for shaded region (replace with actual if available)
    y_err = data['Total Reward'] * 0.1  # 10% of the reward for illustration

    plt.plot(data['Total Process Time'], data['Total Reward'], label=labels[i], linestyle=styles[i], color=colors[i])
    plt.fill_between(data['Total Process Time'],
                     data['Total Reward'] - y_err,
                     data['Total Reward'] + y_err,
                     color=colors[i], alpha=0.2)

# Customize plot aesthetics
plt.xlabel('Process Time (seconds)')
plt.ylabel('Reward')
plt.title('Process Time vs Reward for Different Workers')
plt.legend(loc='upper left', frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()
