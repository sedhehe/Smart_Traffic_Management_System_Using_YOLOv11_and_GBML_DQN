import matplotlib.pyplot as plt
import os

def plot_metrics(avg_queue, avg_wait, total_rewards, episodes, time):
    output_dir = f'plots/{time}'
    os.makedirs(output_dir, exist_ok=True)

    # Plot average queue length
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(avg_queue)), avg_queue, label="Avg Queue Length")
    plt.xlabel("Episodes")
    plt.ylabel("Avg Queue Length")
    plt.title(f"Average Queue Length over {episodes} Episodes")
    plt.legend()
    plt.savefig(f'{output_dir}/avg_queue_{episodes}.png')
    plt.close()

    # Plot average waiting time
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(avg_wait)), avg_wait, label="Avg Wait Time", color='orange')
    plt.xlabel("Episodes")
    plt.ylabel("Avg Wait Time (s)")
    plt.title(f"Average Waiting Time over {episodes} Episodes")
    plt.legend()
    plt.savefig(f'{output_dir}/avg_wait_{episodes}.png')
    plt.close()

    # Plot total rewards
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(total_rewards)), total_rewards, label="Total Rewards", color='green')
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.title(f"Total Rewards over {episodes} Episodes")
    plt.legend()
    plt.savefig(f'{output_dir}/total_rewards_{episodes}.png')
    plt.close()
