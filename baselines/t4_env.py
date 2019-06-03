import sys
sys.path.append('/home/dan/serpent/common')

from tf_agent import T4TFEnv

import gym
import numpy as np

from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines import ACER

import os
import cv2
import gym
from gym import spaces

EP_LENGTH = 100

class ObsState:
    def __init__(self, dir, file):
        split = file.split('_')
        self.file = file
        self.action = split[1]
        self.reward = float(split[2].strip('.jpg'))
        self.observation = cv2.imread(os.path.join(dir, file))[:95,:]

class T4HistoryEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dir):
        super(T4HistoryEnv, self).__init__()
        
        self.action_space = spaces.Discrete(2)
        self.dir = dir
        self.files = os.listdir(dir)
        
        first = cv2.imread(os.path.join(dir, self.files[0]))
        first = first[:95,:]
        self.observation_space = spaces.Box(low=0, high=255, shape=first.shape, dtype=np.uint8)
        
        self.x = 0
        self.state = ObsState(dir, self.files[0])
        
        self.wins = 0
        self.action_count = 0
        
        
    def reset(self):
        return self.state.observation
        
    def step(self, action):
        
        self.action_count += 1
        
        if action == self.state.action:
            reward = self.state.reward
        else:
            if self.state.reward > 0:
                reward = -1.0
            else:
                reward = 1.0
        
        self.x += 1
        if self.x >= len(self.files):
            self.x = 0
        
        self.state = ObsState(self.dir, self.files[self.x])
        
        if reward > 0:
            self.wins += 1
            
        if self.x % 100 == 0:
            print('---')
            print('%d / %d - %d' % (self.wins, self.action_count, int((self.wins / self.action_count) * 100)))    
        
        done = False
        if self.x % 100 == 0:
            done = True
        
        return self.state.observation, reward, done, {}
    
    def render(self, mode='human', close=False):
        pass
        

class T4Env(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, metrics_key='017'):
    super(T4Env, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    
    self._env = T4TFEnv(metrics_key=metrics_key)

    self.action_space = spaces.Discrete(2)
#     self.action_space = spaces.Box(low=-2, high=2, shape=(1,), dtype=np.int8)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=self._env.state_shape, dtype=np.uint8)
  def step(self, action):
    # print('action')  
    # print((self._env.action_space.n / 2) + action[0])
#     action_int = int(action[0])
#     new_action = action_int
#     if action_int is -1:
#         new_action = 2
        
    
    # print(new_action)

    result = self._env._step(int(np.abs(action)))
    return result.observation, result.reward, False, {}
    
  def reset(self):
    print('reset')
    return self._env._reset().observation

  def render(self, mode='human', close=False):
    pass
