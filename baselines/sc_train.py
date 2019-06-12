import sys
sys.path.append('/home/dan/serpent/common')

from tf_agent import T4TFEnv

import gym
import numpy as np

from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
from stable_baselines import A2C
from stable_baselines import PPO2, TRPO
from stable_baselines import ACER
import os
from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds

import gym
from gym import spaces
from t4_env import SCSimEnv

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

log_dir = "./tmp/gym/sac3"
os.makedirs(log_dir, exist_ok=True)
dir = '/home/dan/serpent/frames'

num_workers = 8
model_file = ‘sp500_ppo3_pretrain’

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = SCSimEnv(dir, debug=False, limit=3, rank=rank, total=num_workers, lstm=True)
        #env = gym.make('Pong-v0')
        env.rank = rank
        env.seed(seed + rank)
        
        #env = Monitor(env, log_dir, allow_early_resets=True)
        return env
    set_global_seeds(seed)
    return _init

env = SubprocVecEnv([make_env('', i) for i in range(num_workers)])
#env = Monitor(env, log_dir, allow_early_resets=True)

#env = Monitor(SCSimEnv(dir, continuous=True, debug=False, limit=3, rank=0, total=1), log_dir, allow_early_resets=True)
#env = DummyVecEnv([lambda: SCSimEnv(dir, rank=0, total=1)])

#model = PPO2('CnnPolicy', env, verbose=1, tensorboard_log="./tmp/gym/board/")
model = PPO2.load(model_file)
# model = SAC.load('tmp/gym/boardbest_model')
model.set_env(env)
model.learn(total_timesteps=int(20e6), log_interval=10) #, callback=callback)

model.save(model_file)
