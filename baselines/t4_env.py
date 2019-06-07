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
from gym.envs.registration import EnvSpec
import os
import cv2
import gym
from gym import spaces
from random import shuffle

EP_LENGTH = 100

class ObsState:
    def __init__(self, dir, file):
        file_name = os.path.basename(file)
        split = file_name.split('_')
        self.file = file
        
        if len(split) < 2:
            print(file_name)
            print(split)
            print(self.x)
            print('------')
            
            
        self.action = split[1]
        self.reward = float(split[2].strip('.jpg'))
        
        im = cv2.imread(file)[:,103:]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).reshape((im.shape[0], im.shape[1], -1))
        self.observation = im

def process_dir(path):
    files = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir():
                files = files + process_dir(entry.path)
            else:
                files.append(entry.path)
                
    return files

class T4HistoryEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dir, continuous_action=False):
        super(T4HistoryEnv, self).__init__()
        
        if continuous_action:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int8)
        else:
            self.action_space = spaces.Discrete(3)
        self.dir = dir
        
        files = process_dir(dir)
                
        self.files = files
        shuffle(self.files)
        #first = first[:95,:]
        self.state = ObsState(dir, self.files[0])
        
        self.observation_space = spaces.Box(low=0, high=255, shape=self.state.observation.shape, dtype=np.uint8)
        print(self.state.observation.shape)
        self.x = 0
        
        self.wins = 0
        self.episode_wins = 0
        self.action_count = 0
        
        self.spec = EnvSpec(id='T4History-v0', nondeterministic=True)
        
        self.reward_range = (-1, 1)
        self.continuous_action = continuous_action
        
    def get_next_state(self):
        self.x += 1
        if self.x >= len(self.files):
            shuffle(self.files)
            self.x = 0
        
        while not self.files[self.x].endswith('jpg'):
            self.x += 1
            
        state = ObsState(self.dir, self.files[self.x])
        
        while state.action == 2:
            self.x += 1
            if self.x >= len(self.files):
                self.x = 0
        
            state = ObsState(self.dir, self.files[self.x])
        return state
        
    def reset(self):
        return self.state.observation
        
    def step(self, action):
        
        if (self.continuous_action):
            if action > 0:
                action = 1
            else:
                action = 0
        
        if action == 2:
            reward = 0.0
        else:
            self.action_count += 1
        
            if action == self.state.action:
                reward = self.state.reward
            else:
                if self.state.reward > 0:
                    reward = -1.0
                else:
                    reward = 1.0
        
#         reward = -1.0
        self.state = self.get_next_state()
        
        if reward > 0:
            self.wins += 1
            self.episode_wins += 1
        
        
        if self.x % 10000 == 0 and self.action_count != 0:
            print('---')
            print('%d / %d - %d' % (self.wins, self.action_count, int((self.wins / self.action_count) * 100)))    
            print('%d / %d - %d' % (self.episode_wins, 100, int((self.episode_wins / 100) * 100)))
            
        
        done = False
        if self.x % 100 == 0:
            self.episode_wins = 0
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
