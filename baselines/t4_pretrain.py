from stable_baselines import ACER, PPO2, SAC, A2C
from stable_baselines.gail import ExpertDataset
from t4_env import T4HistoryEnv
from stable_baselines.common.vec_env import DummyVecEnv
import numpy as np
import os
from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds

# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset
'''
dataset = ExpertDataset(expert_path='t4_dataset.npz',
                        traj_limitation=1, batch_size=1)

env = DummyVecEnv([lambda: T4Env()])

model = ACER('CnnPolicy', env, verbose=1) #, nminibatches=1)
# Pretrain the PPO2 model
model.pretrain(dataset, n_epochs=1000)

model.save('acer-t4-pretrain')
'''


best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy training performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
  n_steps += 1
  return True

log_dir = "./tmp/gym/board"
os.makedirs(log_dir, exist_ok=True)

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = T4HistoryEnv(dir)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

dir = '/home/dan/serpent/market_history/sp500'

# env = T4HistoryEnv(dir, continuous_action=True)
# env2 = T4HistoryEnv(dir)
# file_count = len(env.files)
# print(len(env.files))

# env = DummyVecEnv([lambda: env])
# env = Monitor(env, log_dir, allow_early_resets=True)
# env2 = Monitor(env2, log_dir, allow_early_resets=True)\
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
# env = DummyVecEnv([lambda: env])

env = SubprocVecEnv([make_env('', i) for i in range(8)])
# model = PPO2('CnnPolicy', env, verbose=1, tensorboard_log="./tmp/gym/board/")
model = PPO2.load('sp500_ppo2_pretrain')
model.set_env(env)
model.learn(total_timesteps=int(10e6), log_interval=10, callback=callback)

model.save('sp500_ppo2_pretrain')
