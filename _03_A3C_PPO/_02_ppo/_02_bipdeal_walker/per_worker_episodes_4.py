import os
import pandas as pd
import matplotlib.pyplot as plt

# Define directory paths and settings
base_dir = "train_data"  # Update this with your base directory path
worker_folders = {"w_1": 1, "w_4": 4, "w_8": 8}  # Worker configurations and their counts
run_indices = ["_2", "_3"]  # Different run indexes
file_limit = 5

# Initialize data containers
worker_episodes = {}  # To store total episodes for each worker configuration

# Process files by worker amount and run index
for folder, worker_count in worker_folders.items():
    folder_path = os.path.join(base_dir, folder)

    # Collect first five files for each worker count with specified run indexes
    matched_files = sorted([
        f for f in os.listdir(folder_path)
        if any(f.endswith(f"{run}.csv") for run in run_indices)
    ])[:file_limit]

    # Debug print to verify matched files
    print(f"Processing files for {folder}: {matched_files}")

    # Initialize total episodes based on row count
    total_episodes = 0
    for file in matched_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # Strip any whitespace from column names
        df.columns = df.columns.str.strip()

        # Count the number of rows in the DataFrame, assuming each row is one episode
        total_episodes += len(df)

    # Store the total episodes for each worker configuration
    worker_episodes[worker_count] = total_episodes

# Plotting
plt.figure(figsize=(10, 6))

# Extract worker counts and total episodes for plotting
worker_counts = list(worker_episodes.keys())
total_episodes = list(worker_episodes.values())

plt.bar(worker_counts, total_episodes, color=['blue', 'green', 'orange'])

# Customize plot
plt.xlabel('Number of Workers')
plt.ylabel('Total Episodes Processed')
plt.title('Total Episodes Processed by Different Worker Amounts')
plt.xticks(worker_counts, [f'{w} Workers' for w in worker_counts])
plt.grid(axis='y')
plt.tight_layout()
plt.show()
