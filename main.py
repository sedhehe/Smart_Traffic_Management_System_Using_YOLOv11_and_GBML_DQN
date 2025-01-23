from absl import app
from absl import flags
from environment.env import SumoEnv
from agents.dqn import DqnAgent
from replay import ReplayBuffer
import torch
from datetime import datetime
import math
from plots import plot_metrics

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('skip_range', 10, 'time(seconds) range for skip randomly at the beginning')
flags.DEFINE_float('simulation_time', 10000, 'time for simulation')
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
flags.DEFINE_integer('delta_rs_update_time', 10, 'time interval for calculating reward')
flags.DEFINE_string('reward_fn', 'choose-min-waiting-time', 'Reward function to use')
flags.DEFINE_string('net_file', 'nets/2way-single-intersection/single-intersection.net.xml', 'Path to the network file')
flags.DEFINE_string('route_file', 'nets/2way-single-intersection/single-intersection-vhvh.rou.xml', 'Path to the route file')
flags.DEFINE_bool('use_gui', False, 'Use sumo-gui instead of sumo')
flags.DEFINE_integer('num_episodes', 601, 'Number of episodes for training/testing')
flags.DEFINE_string('network', 'dqn', 'Type of network (e.g., dqn)')
flags.DEFINE_string('mode', 'train', 'Mode: train or test')
flags.DEFINE_float('eps_start', 1.0, 'Starting epsilon for epsilon-greedy strategy')
flags.DEFINE_float('eps_end', 0.1, 'Final epsilon for epsilon-greedy strategy')
flags.DEFINE_integer('eps_decay', 83000, 'Decay factor for epsilon')
flags.DEFINE_integer('target_update', 3000, 'Target network update frequency')
flags.DEFINE_string('network_file', '', 'Path to the pre-trained network file')
flags.DEFINE_float('gamma', 0.95, 'Discount factor for future rewards')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
flags.DEFINE_bool('use_sgd', True, 'Use SGD optimizer (True) or RMSprop (False)')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Generate timestamp for saving models and metrics
time = datetime.now().strftime('%Y%m%d')


def main(argv):
    del argv  # Unused arguments

    # Initialize the SUMO environment
    env = SumoEnv(
        net_file=FLAGS.net_file,
        route_file=FLAGS.route_file,
        skip_range=FLAGS.skip_range,
        simulation_time=FLAGS.simulation_time,
        yellow_time=FLAGS.yellow_time,
        delta_rs_update_time=FLAGS.delta_rs_update_time,
        reward_fn=FLAGS.reward_fn,
        mode=FLAGS.mode,
        use_gui=FLAGS.use_gui,
    )

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=20000)

    # Initialize agent
    agent = None
    if FLAGS.network == 'dqn':
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        agent = DqnAgent(
            FLAGS.mode,
            replay_buffer,
            FLAGS.target_update,
            FLAGS.gamma,
            FLAGS.use_sgd,
            FLAGS.eps_start,
            FLAGS.eps_end,
            FLAGS.eps_decay,
            input_dim,
            output_dim,
            FLAGS.batch_size,
            FLAGS.network_file
        )

    # Start training or testing loop
    for episode in range(FLAGS.num_episodes):
        initial_state = env.reset()
        done = False
        invalid_action = False
        total_reward = 0  # Track total reward for the episode

        while not done:
            # Compute the current state
            state = env.compute_state()

            # Select action using the agent
            action = agent.select_action(state, replay_buffer.steps_done, invalid_action)

            # Take a step in the environment
            next_state, reward, done, info = env.step(action)

            if info['do_action'] is None:
                invalid_action = True
                continue
            invalid_action = False

            # Ensure reward is not None
            reward = reward if reward is not None else 0
            total_reward += reward

            # Add experience to replay buffer and train the agent
            if FLAGS.mode == 'train':
                action_scalar = info['do_action'] if isinstance(info['do_action'], int) else info['do_action'].item()
                replay_buffer.add(state, action_scalar, next_state, reward)

                if not agent.update_gamma:
                    agent.learn()
                else:
                    agent.learn_gamma()

        # Print episode statistics
        print(f'Episode: {episode}, Total Reward: {total_reward}')
        print(f'Current Epsilon: {agent.epsilon}')
        print(f'Learn Steps: {agent.learn_steps}, Gamma: {agent.gamma}')

        # Save model weights periodically
        if FLAGS.mode == 'train' and episode % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f'weights/weights_{time}_episode_{episode}.pth')

        # Plot metrics periodically
        if FLAGS.mode == 'train' and episode % 100 == 0:
            plot_metrics(env.avg_queue, env.avg_wait, env.total_rewards, episode, time)

    # Close the environment after all episodes
    env.close()


if __name__ == '__main__':
    app.run(main)
