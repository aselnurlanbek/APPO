import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Base directory path for your data
base_dir = "train_data"

# Define worker configurations
worker_folders = {"w_1": 1, "w_2": 2, "w_4": 4}
worker_colors = {"w_1": "blue", "w_2": "green", "w_4": "orange"}

# Data containers
worker_data_reward = {}
worker_data_process_time = {}

# Function to convert time format (HH:MM:SS) to seconds
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s

# Smoothing function
def smooth_data(data, window_size=10):
    return data.rolling(window=window_size, min_periods=1).mean()

# Process data for Episode vs Reward
for folder in worker_folders:
    folder_path = os.path.join(base_dir, folder)
    matched_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    data_frames = [pd.read_csv(os.path.join(folder_path, file)) for file in matched_files]
    combined_data = pd.concat(data_frames).groupby("Episode").mean(numeric_only=True)
    combined_data["Episode Reward"] = smooth_data(combined_data["Episode Reward"], window_size=10)
    worker_data_reward[folder] = combined_data

# Process data for Process Time vs Reward
for worker in worker_folders:
    worker_dir = os.path.join(base_dir, worker)
    matched_files = [f for f in os.listdir(worker_dir) if f.startswith("master_processor") and f.endswith(".csv")]
    data_frames = []
    for file in matched_files:
        df = pd.read_csv(os.path.join(worker_dir, file))
        df["Total Process Time"] = df["Total Process Time"].apply(time_to_seconds)
        data_frames.append(df)
    combined_data = pd.concat(data_frames).groupby("Total Process Time").mean()
    combined_data["Total Reward"] = smooth_data(combined_data["Total Reward"], window_size=10)
    worker_data_process_time[worker] = combined_data

# Create a single figure with two subplots arranged horizontally
fig, axs = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('LunarLanderContinuous-v3', fontsize=16, fontweight='bold')

# Subplot 1: Smoothed Episode vs Reward
for folder, data in worker_data_reward.items():
    reward_mean = data["Episode Reward"]
    reward_std = data["Episode Reward"].std()
    axs[0].plot(data.index, reward_mean, label=f'{worker_folders[folder]} Workers', color=worker_colors[folder], linestyle='-')
    axs[0].fill_between(data.index, reward_mean - reward_std, reward_mean + reward_std, color=worker_colors[folder], alpha=0.2)

axs[0].set_title('Smoothed Episode vs Reward')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Reward')
axs[0].legend()
axs[0].grid(True)

# Subplot 2: Smoothed Process Time vs Reward
for worker, data in worker_data_process_time.items():
    reward_mean = data["Total Reward"]
    reward_std = data["Total Reward"].std()
    axs[1].plot(data.index, reward_mean, label=f'{worker_folders[worker]} Workers', color=worker_colors[worker], linestyle='--')
    axs[1].fill_between(data.index, reward_mean - reward_std, reward_mean + reward_std, color=worker_colors[worker], alpha=0.2)

axs[1].set_title('Smoothed Process Time vs Reward')
axs[1].set_xlabel('Process Time (seconds)')
axs[1].set_ylabel('Reward')
axs[1].legend()
axs[1].grid(True)

# Adjust layout and display the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the title
plt.show()
