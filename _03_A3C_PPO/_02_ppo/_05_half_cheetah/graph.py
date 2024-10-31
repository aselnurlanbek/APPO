import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os

from b_actor_and_critic import CSV_DIR

# Fetch all CSV files matching the pattern
csv_files = glob.glob(f"{CSV_DIR}/appo_HalfCheetah-v4_*_workers_*.csv")

# Prepare a dictionary to store data for each worker
worker_data = {}

# Regular expression to extract the worker number and environment name
worker_pattern = re.compile(r"appo_(.*)_.*_workers_(\d+).csv$")

# Extract environment name (assumes all files have the same env)
env_name = None

# Read each CSV file
for csv_file in csv_files:
    # Extract environment and worker identifiers using regex
    match = worker_pattern.search(csv_file)
    if match:
        if env_name is None:
            env_name = match.group(1)  # Get the environment name
        worker_name = match.group(2)  # Get the worker number

        # Read CSV data
        df = pd.read_csv(csv_file, names=[
            "Validation Episode Reward Average", "Training Episode Reward",
            "Policy Loss", "Critic Loss", "Average Mu", "Average Std",
            "Average Action", "Training Episode", "Training Steps"
        ], skiprows=1)

        # If the worker count is 4, select every 4th element and adjust step sizes
        if int(worker_name) == 4:
            df = df.iloc[::4, :].reset_index(drop=True)
            # Reduce the Training Episode values by dividing them by 4
            df["Training Episode"] = df["Training Episode"] // 4

            # Save the filtered data to a new CSV file
            new_csv_file = os.path.join(CSV_DIR, f"filtered_ppo_{env_name}_workers_4.csv")
            df.to_csv(new_csv_file, index=False)
            print(f"Filtered CSV saved to: {new_csv_file}")

        # Store in the dictionary
        worker_data[worker_name] = df

# Create the plot
plt.figure(figsize=(12, 8))

# Define a new set of distinct colors and line styles
colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFF5', '#FF8C33', '#8CFF33', '#FF3333', '#3333FF']
line_styles = ['-', '--', '-.', ':']

# Plot data for each worker
for i, (worker_name, df) in enumerate(worker_data.items()):
    # Apply rolling mean for smoothing (window size = 5, adjust as needed)
    smoothed_rewards = df["Validation Episode Reward Average"].rolling(window=5).mean()
    plt.plot(df["Training Episode"], smoothed_rewards,
             label=f"Worker {worker_name}", color=colors[i % len(colors)],
             linestyle=line_styles[i % len(line_styles)], linewidth=1.5)

# Add title and labels
plt.title(f"{env_name} - PPO Training: Validation Episode Reward by Worker", fontsize=14)
plt.xlabel("Training Episode", fontsize=12)
plt.ylabel("Validation Episode Reward Average (Smoothed)", fontsize=12)
plt.legend(title="Workers", fontsize=10, title_fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Show the plot
plt.show()
