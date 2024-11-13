import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Base directory path for your data
base_dir = "train_data"

# Define the worker configurations and parameters from each file
worker_folders = {"w_1": 1, "w_4": 4, "w_8": 8}
run_indices = ["_2", "_3"]
file_limit = 5

# Color mapping for workers
worker_colors = {
    "w_1": "blue",
    "w_4": "green",
    "w_8": "orange"
}

# Initialize data containers
worker_episodes = {}
worker_data_reward = {}
worker_data_process_time = {}
worker_data_train_steps = {}

# 1. Data processing for total episodes by worker count
for folder, worker_count in worker_folders.items():
    folder_path = os.path.join(base_dir, folder)
    matched_files = sorted([f for f in os.listdir(folder_path) if any(f.endswith(f"{run}.csv") for run in run_indices)])[:file_limit]
    total_episodes = sum([len(pd.read_csv(os.path.join(folder_path, file))) for file in matched_files])
    worker_episodes[worker_count] = total_episodes

# 2. Data processing for Episode vs Reward
for folder in worker_folders:
    folder_path = os.path.join(base_dir, folder)
    matched_files = sorted([f for f in os.listdir(folder_path) if any(f.endswith(f"{run}.csv") for run in run_indices)])[:file_limit]
    data_frames = [pd.read_csv(os.path.join(folder_path, file)) for file in matched_files]
    combined_data = pd.concat(data_frames).groupby(level=0).mean(numeric_only=True)
    worker_data_reward[folder] = combined_data

# 3. Data processing for Process Time vs Reward
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s

for worker in worker_folders:
    worker_dir = os.path.join(base_dir, worker)
    matched_files = sorted([f for f in os.listdir(worker_dir) if f.startswith("master_processor") and f.endswith(".csv")])[:file_limit]
    data_frames = []
    for file in matched_files:
        df = pd.read_csv(os.path.join(worker_dir, file))
        df['Total Process Time'] = df['Total Process Time'].apply(time_to_seconds)
        data_frames.append(df)
    combined_data = pd.concat(data_frames).groupby(level=0).mean()
    worker_data_process_time[worker] = combined_data

# 4. Data processing for Training Steps vs Reward
smoothing_window = 10
worker_counts = ["_1", "_4", "_8"]

for count in worker_counts:
    data_frames = []
    for folder in worker_folders:
        folder_path = os.path.join(base_dir, folder)
        matched_files = sorted([f for f in os.listdir(folder_path) if f.endswith(f"workers{count}.csv")])[:file_limit]
        data_frames.extend([pd.read_csv(os.path.join(folder_path, file)) for file in matched_files])
    combined_data = pd.concat(data_frames).groupby(level=0).mean()
    combined_data['Training Steps'] = combined_data['Training Steps'].rolling(window=smoothing_window).mean()
    combined_data['Validation Episode Reward Average'] = combined_data['Validation Episode Reward Average'].rolling(window=smoothing_window).mean()
    worker_data_train_steps[count] = combined_data

# Plotting all graphs together with a shared title
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Hopper-v5', fontsize=16, fontweight='bold')

# Plot 1: Total Episodes by Worker Count
bars = axs[0, 0].bar(worker_episodes.keys(), worker_episodes.values(), color=[worker_colors[f"w_{k}"] for k in worker_episodes.keys()])
# axs[0, 0].bar(worker_episodes.keys(), worker_episodes.values(), color=[worker_colors[f"w_{k}"] for k in worker_episodes.keys()])
axs[0, 0].set_title('Total Episodes by Worker Count')
axs[0, 0].set_xlabel('Number of Workers')
axs[0, 0].set_ylabel('Total Episodes Processed')
# axs[0, 0].legend([f"{k} Workers" for k in worker_episodes.keys()], loc="upper left")

axs[0, 0].set_xticks([1, 4, 8])

# Create a legend for the first plot with specific color descriptions
legend_labels = [f"{worker_folders[f'w_{count}']} Workers" for count in worker_episodes.keys()]
axs[0, 0].legend(bars, legend_labels, loc="upper left")

# Plot 2: Episode vs Reward
for folder, data in worker_data_reward.items():
    axs[0, 1].plot(data['Episode'], data['Episode Reward'], label=f'{worker_folders[folder]} Workers', color=worker_colors[folder])
axs[0, 1].set_title('Episode vs Reward')
axs[0, 1].set_xlabel('Episode')
axs[0, 1].set_ylabel('Reward')
axs[0, 1].legend()

# Plot 3: Process Time vs Reward
for worker, data in worker_data_process_time.items():
    y_err = data['Total Reward'] * 0.1  # Sample error
    axs[1, 0].plot(data['Total Process Time'], data['Total Reward'], label=f'{worker_folders[worker]} Workers', color=worker_colors[worker])
    axs[1, 0].fill_between(data['Total Process Time'], data['Total Reward'] - y_err, data['Total Reward'] + y_err, color=worker_colors[worker], alpha=0.2)
axs[1, 0].set_title('Process Time vs Reward')
axs[1, 0].set_xlabel('Process Time (seconds)')
axs[1, 0].set_ylabel('Reward')
axs[1, 0].legend()

# Plot 4: Training Steps vs Reward (Smoothed)
for count, data in worker_data_train_steps.items():
    worker_key = f"w_{count.strip('_')}"
    axs[1, 1].plot(data['Training Steps'], data['Validation Episode Reward Average'], label=f'{worker_folders[worker_key]} Workers', color=worker_colors[worker_key])
axs[1, 1].set_title('Training Steps vs Reward (Smoothed)')
axs[1, 1].set_xlabel('Training Steps')
axs[1, 1].set_ylabel('Reward')
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for the title
plt.show()
