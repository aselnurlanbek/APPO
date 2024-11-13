import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# Function to retrieve the last n files in a folder matching a specific pattern
def get_last_n_files(folder, pattern, n=5):
    files = sorted(glob(os.path.join(folder, pattern)), key=os.path.getmtime, reverse=True)
    return files[:n]

# Function to plot averaged data from a list of dataframes
def plot_average(dataframes, x_col, y_col, title, xlabel, ylabel):
    """Utility function to plot averaged data from a list of dataframes."""
    concatenated_df = pd.concat(dataframes)
    averaged_df = concatenated_df.groupby(x_col).mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(averaged_df[x_col], averaged_df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Define base path for data
base_path = 'train_data'
folders = ['w_1', 'w_4', 'w_8']  # Folder structure

# 1. Wall time (process time) vs reward
process_time_rewards = []
for folder in folders:
    files = get_last_n_files(os.path.join(base_path, folder), "master_processor_*.csv")
    dataframes = [pd.read_csv(f) for f in files]
    process_time_rewards.extend(dataframes)

plot_average(
    process_time_rewards,
    x_col="Total Process Time",
    y_col="Total Reward",
    title="Wall Time vs Reward",
    xlabel="Wall Time (Process Time)",
    ylabel="Reward"
)

# 2. Total episode vs reward per worker
worker_rewards = []
for folder in folders:
    for worker in range(3):  # Assuming multiple workers (adjust range as needed)
        files = get_last_n_files(os.path.join(base_path, folder), f"worker_{worker}_metrics_*.csv")
        dataframes = [pd.read_csv(f) for f in files]
        worker_rewards.extend(dataframes)

plot_average(
    worker_rewards,
    x_col="Episode",
    y_col="Episode Reward",
    title="Total Episode vs Reward per Worker",
    xlabel="Total Episode",
    ylabel="Reward"
)

# 3. Total training steps vs reward
training_step_rewards = []
for folder in folders:
    files = get_last_n_files(os.path.join(base_path, folder), "appo_HalfCheetah-v5_*.csv")
    dataframes = [pd.read_csv(f) for f in files]
    training_step_rewards.extend(dataframes)

plot_average(
    training_step_rewards,
    x_col="Training Steps",
    y_col="[TRAIN] Episode Reward",
    title="Total Training Steps vs Reward",
    xlabel="Total Training Steps",
    ylabel="Reward"
)

# 4. Per worker episodes vs reward
plot_average(
    worker_rewards,
    x_col="Episode",
    y_col="Episode Reward",
    title="Per Worker Episodes vs Reward",
    xlabel="Per Worker Episodes",
    ylabel="Reward"
)
