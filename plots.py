import matplotlib.pyplot as plt


def plot_average_queue(avg_q, episode, time):
    plt.figure(1)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Average Queue')
    plt.plot(avg_q)
    plt.savefig('record/queue_{0}_episodes{1}.png'.format(time, episode))

def plot_average_waiting_times(avg_waiting_times, episode, time):
    plt.figure(2)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Average Waiting Time')
    plt.plot(avg_waiting_times)
    plt.savefig('record/avg_waiting_times_{0}_episodes{1}.png'.format(time, episode))

def plot_total_rewards(total_rewards, episode, time):
    plt.figure(3)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')
    plt.plot(total_rewards)
    plt.savefig('record/total_rewards_{0}_episodes{1}.png'.format(time, episode))
