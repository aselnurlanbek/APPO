import os
import pandas as pd
import matplotlib.pyplot as plt

# Define worker folders and file paths
base_dir = "train_data"  # Update this with your base directory path
worker_folders = ["w_1", "w_4", "w_8"]  # Different worker amounts

plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'orange']
labels = ['1 Worker', '4 Workers', '8 Workers']

# Process and plot data for each worker configuration
for i, folder in enumerate(worker_folders):
    files = [os.path.join(base_dir, folder, f) for f in os.listdir(os.path.join(base_dir, folder)) if
             f.startswith("wall_time_data")]  # Adjust pattern if needed
    data_frames = [pd.read_csv(file) for file in files[:5]]

    # Combine data and average if needed
    combined_data = pd.concat(data_frames).groupby(level=0).mean()

    plt.plot(combined_data['Wall Time'], combined_data['Episode Reward'], label=labels[i], color=colors[i])

plt.xlabel('Wall Time')
plt.ylabel('Reward')
plt.title('Wall Time vs Reward for Different Worker Amounts')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


#2222
plt.figure(figsize=(8, 6))

# Process and plot data for each worker configuration
for i, folder in enumerate(worker_folders):
    files = [os.path.join(base_dir, folder, f) for f in os.listdir(os.path.join(base_dir, folder)) if
             f.startswith("episode_data")]  # Adjust pattern if needed
    data_frames = [pd.read_csv(file) for file in files[:5]]

    # Combine data and average if needed
    combined_data = pd.concat(data_frames).groupby(level=0).mean()

    plt.plot(combined_data['Total Episode'], combined_data['Episode Reward'], label=labels[i], color=colors[i])

plt.xlabel('Total Episode')
plt.ylabel('Reward')
plt.title('Total Episode vs Reward for Different Worker Amounts')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


#3
plt.figure(figsize=(8, 6))

# Process and plot data for each worker configuration
for i, folder in enumerate(worker_folders):
    files = [os.path.join(base_dir, folder, f) for f in os.listdir(os.path.join(base_dir, folder)) if f.startswith("training_steps_data")]  # Adjust pattern if needed
    data_frames = [pd.read_csv(file) for file in files[:5]]

    # Combine data and average if needed
    combined_data = pd.concat(data_frames).groupby(level=0).mean()

    plt.plot(combined_data['Total Training Steps'], combined_data['Episode Reward'], label=labels[i], color=colors[i])

plt.xlabel('Total Training Steps')
plt.ylabel('Reward')
plt.title('Total Training Steps vs Reward for Different Worker Amounts')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#  444
plt.figure(figsize=(8, 6))

# Process and plot data for each worker configuration
for i, folder in enumerate(worker_folders):
    files = [os.path.join(base_dir, folder, f) for f in os.listdir(os.path.join(base_dir, folder)) if
             f.startswith("training_steps_data")]  # Adjust pattern if needed
    data_frames = [pd.read_csv(file) for file in files[:5]]

    # Combine data and average if needed
    combined_data = pd.concat(data_frames).groupby(level=0).mean()

    plt.plot(combined_data['Total Training Steps'], combined_data['Episode Reward'], label=labels[i], color=colors[i])

plt.xlabel('Total Training Steps')
plt.ylabel('Reward')
plt.title('Total Training Steps vs Reward for Different Worker Amounts')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



