from absl import app
from absl import flags
from environment.env import SumoEnv
from agents.dqn import DqnAgent
from replay import ReplayBuffer
import torch
from datetime import datetime
import math
from plots import plot_average_queue, plot_average_waiting_times, plot_total_rewards

FLAGS = flags.FLAGS
flags.DEFINE_integer('skip_range', 10, 'time(seconds) range for skip randomly at the beginning')
flags.DEFINE_float('simulation_time', 10000, 'time for simulation')
flags.DEFINE_integer('yellow_phase_duration', 2, 'time for yellow phase')
flags.DEFINE_integer('reward_update_interval', 10, 'time for calculate reward')
flags.DEFINE_string('reward_function', 'choose-min-waiting-time', '')
flags.DEFINE_string('network_file', 'nets/2way-single-intersection/single-intersection.net.xml', '')
flags.DEFINE_string('route_file', 'nets/2way-single-intersection/single-intersection-vhvh.rou.xml', '')
flags.DEFINE_bool('use_gui', True, 'use sumo-gui instead of sumo')
flags.DEFINE_integer('num_episodes', 1, '')
flags.DEFINE_string('network_type', 'dqn', '')
flags.DEFINE_string('mode', 'eval', '')
flags.DEFINE_float('epsilon_start', 1.0, '')
flags.DEFINE_float('epsilon_end', 0.1, '')
flags.DEFINE_integer('epsilon_decay', 83000, '')
flags.DEFINE_integer('target_update_interval', 3000, '')
flags.DEFINE_string('pretrained_weights', 'weights/weights_20250124_400.pth', '')
flags.DEFINE_float('gamma', 0.95, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_bool('use_sgd', True, 'Training with the optimizer SGD or RMSprop')

device = "cuda" if torch.cuda.is_available() else "cpu"

current_date = str(datetime.now()).split('.')[0].split(' ')[0].replace('-', '')

def main(argv):
    del argv
    env = SumoEnv(net_file=FLAGS.network_file,
                  route_file=FLAGS.route_file,
                  skip_range=FLAGS.skip_range,
                  simulation_time=FLAGS.simulation_time,
                  yellow_phase_duration=FLAGS.yellow_phase_duration,
                  reward_update_interval=FLAGS.reward_update_interval,
                  reward_function=FLAGS.reward_function,
                  mode=FLAGS.mode,
                  use_gui=FLAGS.use_gui)
    replay_buffer = ReplayBuffer(capacity=20000)

    if FLAGS.network_type == 'dqn':
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        agent = DqnAgent(FLAGS.mode, replay_buffer, FLAGS.target_update_interval, FLAGS.gamma, FLAGS.use_sgd, FLAGS.epsilon_start,
                         FLAGS.epsilon_end, FLAGS.epsilon_decay, input_dim, output_dim, FLAGS.batch_size, FLAGS.pretrained_weights)

    avg_waiting_times = []
    total_rewards = []

    for episode in range(FLAGS.num_episodes):
        initial_state = env.reset()
        env.train_state = initial_state
        done = False
        invalid_action = False
        episode_rewards = 0
        while not done:
            state = env.compute_state()
            action = agent.select_action(state, replay_buffer.steps_done, invalid_action)
            next_state, reward, done, info = env.step(action)
            if info['do_action'] is None:
                invalid_action = True
                continue
            invalid_action = False

            if FLAGS.mode == 'train':
                replay_buffer.add(env.train_state, env.next_state, reward, info['do_action'])
                agent.learn() if not agent.update_gamma else agent.learn_gamma()

            if reward is not None:
                episode_rewards += reward

        avg_waiting_times.append(env.compute_average_waiting_time())
        total_rewards.append(episode_rewards)

        env.close()
        if FLAGS.mode == 'train' and episode % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f'weights/weights_{current_date}_{episode}.pth')

        print(f'i_episode: {episode}')
        print(f'eps_threshold = : {FLAGS.epsilon_end + (FLAGS.epsilon_start - FLAGS.epsilon_end) * math.exp(-1. * replay_buffer.steps_done / FLAGS.epsilon_decay)}')
        print(f'learn_steps: {agent.learn_steps}')
        print(f'gamma: {agent.gamma}')

        if FLAGS.mode == 'train' and episode % 100 == 0:
            plot_average_queue(env.avg_queue, episode, current_date)
            plot_average_waiting_times(avg_waiting_times, episode, current_date)
            plot_total_rewards(total_rewards, episode, current_date)

if __name__ == '__main__':
    app.run(main)