import os
import pandas as pd
import matplotlib.pyplot as plt

# Base directory path for your data
base_dir = "train_data"

# Define worker configurations
worker_folders = {"w_1": 1, "w_2": 2, "w_4": 4}
# Updated color palette
worker_colors = {"w_1": "#1f77b4", "w_2": "#d62728", "w_4": "#2ca02c"}  # Blue, Orange, Green

# Data containers
worker_data_steps_reward = {}
worker_data_process_time_reward = {}

# Function to convert time format (HH:MM:SS) to seconds
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s

# Smoothing function
def smooth_data(data, window_size=10):
    return data.rolling(window=window_size, min_periods=1).mean()

# Process data for Total Time Steps vs Total Reward
for folder in worker_folders:
    folder_path = os.path.join(base_dir, folder)
    matched_files = [f for f in os.listdir(folder_path) if f.startswith("master_processor") and f.endswith(".csv")]
    data_frames = [pd.read_csv(os.path.join(folder_path, file)) for file in matched_files]
    combined_data = pd.concat(data_frames).groupby("Total Time Steps").mean(numeric_only=True)
    combined_data["Total Reward"] = smooth_data(combined_data["Total Reward"], window_size=10)
    worker_data_steps_reward[folder] = combined_data

# Process data for Total Process Time vs Total Reward
for worker in worker_folders:
    worker_dir = os.path.join(base_dir, worker)
    matched_files = [f for f in os.listdir(worker_dir) if f.startswith("master_processor") and f.endswith(".csv")]
    data_frames = []
    for file in matched_files:
        df = pd.read_csv(os.path.join(worker_dir, file))
        df["Total Process Time"] = df["Total Process Time"].apply(time_to_seconds)
        data_frames.append(df)
    combined_data = pd.concat(data_frames).groupby("Total Process Time").mean(numeric_only=True)
    combined_data["Total Reward"] = smooth_data(combined_data["Total Reward"], window_size=10)
    worker_data_process_time_reward[worker] = combined_data

# Determine global y-axis limits with padding
global_min_reward = min(data["Total Reward"].min() for data in worker_data_steps_reward.values())
global_max_reward = max(data["Total Reward"].max() for data in worker_data_steps_reward.values())
padding = 150  # Add padding to prevent clipping
y_min = global_min_reward - padding
y_max = global_max_reward + padding

# Plot 1: Smoothed Total Time Steps vs Total Reward
plt.figure(figsize=(12, 8))
for folder, data in worker_data_steps_reward.items():
    reward_mean = data["Total Reward"]
    reward_std = data["Total Reward"].std()
    plt.plot(data.index, reward_mean, label=f'{worker_folders[folder]} Workers', color=worker_colors[folder], linestyle='-')
    plt.fill_between(data.index, reward_mean - reward_std, reward_mean + reward_std, color=worker_colors[folder], alpha=0.2)

plt.ylim(y_min, y_max)  # Set global y-axis limits
plt.xlabel('Training Time Steps', fontsize=16)
plt.ylabel('Reward', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Plot 2: Smoothed Total Process Time vs Total Reward
plt.figure(figsize=(12, 8))
for worker, data in worker_data_process_time_reward.items():
    reward_mean = data["Total Reward"]
    reward_std = data["Total Reward"].std()
    plt.plot(data.index, reward_mean, label=f'{worker_folders[worker]} Workers', color=worker_colors[worker], linestyle='--')
    plt.fill_between(data.index, reward_mean - reward_std, reward_mean + reward_std, color=worker_colors[worker], alpha=0.2)

plt.ylim(y_min, y_max)  # Set global y-axis limits
plt.xlabel('Process Time (seconds)', fontsize=16)
plt.ylabel('Reward', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
