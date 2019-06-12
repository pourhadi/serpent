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
from tabulate import tabulate
import tableformatter as tf
EP_LENGTH = 100

from serpent.game_agent import GameAgent
# from .ppo import SerpentPPO
from serpent import cv
import tesserocr
from serpent.frame_transformer import FrameTransformer
from PIL import Image
import re
from serpent.input_controller import MouseButton
from serpent.frame_grabber import FrameGrabber
import pickle
# from agent_model import AgentModel
import json
from datetime import datetime
from serpent import serpent
from serpent.input_controller import InputController
from serpent.window_controller import WindowController
from serpent.visual_debugger.visual_debugger import VisualDebugger
from time import sleep
import subprocess
import time

class SCState:
    def draw_trade(self, trade_type, pl_percent):
    
        width = 30
        cv2.rectangle(self.image, (0,0),(width,self.image.shape[0]),(0,0,0),-1)
        
        t_origin = (0,0)
        t_end = (width,int(self.image.shape[0]/2))
        
        if trade_type == 0:
            t_origin = (0,int(self.image.shape[0]/2))
            t_end = (width,0)
            
        cv2.line(self.image,t_origin,t_end,(255,255,255),1)
        
        start_rad = 1
        end_rad = 15
        
        if pl_percent > 1.0: pl_percent = 1.0
        if pl_percent < -1.0: pl_percent = -1.0
        
        rad = int(start_rad + ((end_rad - start_rad) * np.abs(pl_percent)))
        center = (int(width / 2), int(self.image.shape[0]/4 * 3))
        color = (255,255,255)
        border = -1
        if pl_percent < 0:
            border = 1
        cv2.circle(self.image,center,rad,color,border)
        
        
    def __init__(self, dir, file=None, open_trade=False, trade_type=0, pl_percent=0, price_only=False, files=[]):
        self.file = file
        split = file.split('_')
        self.time = int(split[0])
        self.price = float(split[1].strip('.jpg'))
        
        if not price_only:
            '''
            if len(files) > 0:
                imgs = []
                for file in files:
                    im = cv2.imread(os.path.join(dir, file))[8:68,100:190]
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                    imgs.append(im)
                    
                self.image = np.stack(imgs, axis=2)
            else:
            '''
            self.image = cv2.imread(os.path.join(dir, file))[8:100,100:190]
            #self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            #self.image = self.image.reshape((self.image.shape[0], self.image.shape[1], -1))
            
            if open_trade:
                self.draw_trade(trade_type, pl_percent)

def to_pl(current_price, start_price):
    price_diff = current_price - start_price
    price_diff *= 100
    price_diff /= 25
    price_diff *= 12.5
    return price_diff
    
class SCEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dir):
        self.game = serpent.initialize_game('T4TF1')
        self.game.launch()
        self.game.start_frame_grabber()
        
        self.action_space = spaces.Dicrete(2)
        
        self.spec = EnvSpec(id='SCReplayEnv-v0', nondeterministic=True)
        
        self.dir = dir
        
        self.files = os.listdir(dir)
        print(len(self.files))
        self.files.sort()
        self.state = SCState(dir, self.files[0])
        self.x = 0

        self.observation_space = spaces.Box(low=0, high=255, shape=self.state.image.shape, dtype=np.uint8)
        
        self.open_trade = False
        self.trade_price = 0.0
        self.trade_type = 0 # 0 = buy, 1 = sell
        
        self.actions = 0
        self.trades = 0
        
        self.pl = 0
        
        self.rank = 0
        
        self.buys = 0
        self.sells = 0
        self.holds = 0
        self.closes = 0
        self.auto_closes = 0
        self.last_price = 0


    def write_order(self, order_type):
        write_start = time.time()
        with open('/home/dan/.wine/drive_c/input.txt', 'w') as f:
            f.write('%d' % (order_type))
        
        self.step_write_time += (time.time() - write_start)
        self.previous_write = order_type

    def step_forward(self):
        self.write_order(3)
            
    def read_state():
        result = ['','']
        while len(result[0]) < 1 or len(result[1]) < 1:
            with open('/home/dan/.wine/drive_c/output.txt', 'r') as f:
                result = [x.strip() for x in f.read().split(',')]
        
        return (int(result[0]), int(result[1]), float(result[2]), int(result[3]))

    def reset(self):
        return self.state.image


class TradeState:
    def __init__(self, trade_type, price):
        #print('new trade state - %f' % price)
        self.trades = [price]
        self.type = trade_type
    
    def add_position(self, price):
        self.trades.append(price)
        
    def remove_position(self):
        return self.trades.pop(0)
        
    def average_price(self):
        t = 0
        for price in self.trades:
            t += price
        t = t / len(self.trades)
        return t
        
    def count(self):
        return len(self.trades)
        
    def tick_diff(self, from_price):
        p = from_price - self.average_price()
        p = (p * 100) / 25
        
        if self.type == 1:
            p = -p
        return p
        
    def tick_value(self):
        return 12.5 * len(self.trades)

class SCSimEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dir, continuous=False, debug=False, limit=3, rank=0, total=0, lstm=False):
    
        self.continuous = continuous
        self.debug = debug
        self.limit = limit
        
        if continuous:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)
        self.spec = EnvSpec(id='SCSimEnv-v0', nondeterministic=True)
        
        self.dir = dir
        
        self.files = os.listdir(dir)
        print(len(self.files))
        self.files.sort()
        self.x = 100
        self.state = SCState(dir, self.files[0], files=[self.files[i] for i in range(self.x - 4, self.x)])
        self.prev_state = self.state
        
        
        if rank > 0:
            self.x = int((len(self.files) / total) * (rank))
            self.x += 100
        
        if lstm:
            self.x = 5 + rank
            
        while self.state.price == 0.0:
            self.x += 1
            print('inc x')
            self.state = SCState(dir, self.files[self.x], files=[self.files[i] for i in range(self.x - 4, self.x)])
        
        print('start x: %d' % self.x)

        print(self.state.image.shape)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.state.image.shape, dtype=np.uint8)
        
        self.open_trade = False
        self.trade_price = 0.0
        self.trade_type = 0 # 0 = buy, 1 = sell
        
        self.actions = 0
        self.trades = 0
        
        self.pl = 0
        
        self.rank = 0
        
        self.buys = 0
        self.sells = 0
        self.holds = 0
        self.closes = 0
        self.auto_closes = 0
        self.high_pl = -1e100
        self.low_pl = 1e100
        
        self.wins = 0
        self.num_trades = 0
        self.trade_prices = []
        
        self.trade_state = 0
        self.last_x = 0
        #self.reward_range = (-100, 100)
        
    def reset(self):
        self.trades += 1
        self.open_trade = True
        self.trade_price = self.state.price
        return self.state.image 
    
    # no open trade:
        # 0 - buy
        # 1 - sell
        # 2 - hold
    # open trade:
        # 0, 1 - close
        # 2 - hold
        
    def bump_x(self):
        self.x += 1
        if self.x >= len(self.files):
            self.x = 5
            
        state = SCState(self.dir, self.files[self.x], False, 0, 0, price_only=True)
        while state.price == 0.0:
            self.x += 1
            if self.x >= len(self.files):
                self.x = 5
        
            state = SCState(self.dir, self.files[self.x], False, 0, 0, price_only=True)
            
    def step_v2(self, action):
        if self.trade_state != 0:
            if action == self.trade_state.type:
                self.trade_state.add_position(self.state.price)
                
                reward = self.trade_state.tick_diff(self.state.price) * self.trade_state.tick_value()
                
            else:
                if action == 2:
                    reward = self.trade_state.tick_diff(self.state.price) * self.trade_state.tick_value()
                    
                    if np.abs(reward) > 200:
                        self.pl += reward
                        self.trade_state = 0
                else:
                    removed = self.trade_state.remove_position()
                    
                    ticks = ((self.state.price - removed) * 100) / 25                       
                    if self.trade_state.type == 1:
                        ticks = -ticks
                    
                    reward = ticks * 12.5
                    
                    self.pl += reward
                    
                    if self.trade_state.count() == 0:
                        self.trade_state = 0
                        
        else:
            if action == 2:
                reward = 0
            else:
                self.trade_state = TradeState(action, self.state.price)
                reward = 0
                
        self.bump_x()
        
        pl_percent = 0
        min_pl = 0
        max_pl = 200
        trade_type = 0
        
        #print(reward)
        
        
        if self.trade_state != 0:
            #print(self.trade_state.trades)
            #print(self.trade_state.tick_value())
            #print(self.trade_state.average_price())
            diff_for_trade = to_pl(self.state.price, self.trade_state.average_price())
            if self.trade_state.type == 1:
                diff_for_trade = -diff_for_trade
                
            pl_percent = np.abs(diff_for_trade) / max_pl
            
            if diff_for_trade < 0:
                pl_percent = -pl_percent
                
            trade_type = self.trade_state.type
       
        self.state = SCState(self.dir, self.files[self.x], self.trade_state != 0, trade_type, pl_percent)
        
        done = False
        if self.x % 100 == 0:
            done = True
        
        if action == 0:
            self.buys += 1
        elif action == 1:
            self.sells += 1
        else:
            self.holds += 1
            
        if self.x % 1000 == 0:
            cols = ['buys', 'sells', 'holds', 'pl']
            rows = [(self.buys, self.sells, self.holds, self.pl)]
        
            print(tf.generate_table(rows, cols))
        
        #print(self.state.image.shape)
        return self.state.image, reward, done, {}
        
    def step_v3(self, action):
        if action == 0 or action == 1:
            tick_diff = 0
            new_state = self.state
            while tick_diff <= self.limit:
                self.bump_x()
                if self.x == 5:
                    self.state = SCState(self.dir, self.files[self.x], False, 0, 0, files=[self.files[i] for i in range(self.x - 4, self.x)])
                    return self.state.image, 0, False, {}
            
                new_state = SCState(self.dir, self.files[self.x], False, 0, 0, price_only=True)
                tick_diff = np.abs(((self.state.price - new_state.price) * 100) / 25)
                
            price_diff = to_pl(self.state.price, new_state.price)
            if action == 1:
                price_diff = -price_diff
            
            price_diff -= 18.5
            self.pl += price_diff    
            reward = 1 if price_diff > 0 else -1
            if reward > 0:
                self.wins += 1
                
            if action == 0:
                self.buys += 1
            else:
                self.sells += 1
            
        else:
            self.holds += 1
            reward = 0
            
        self.bump_x()
        self.state = SCState(self.dir, self.files[self.x], False, 0, 0, files=[self.files[i] for i in range(self.x - 4, self.x)])
        
        
        done = False
        if self.actions % 100 == 0:
            done = True
        
        
        
        if self.actions % 1000 == 0:
            tt = self.buys + self.sells
            ws = 0            
            if tt > 0:
                ws = int((self.wins / tt) * 100)
            cols = ['x', 'buys', 'sells', 'holds', 'wins %', 'pl']
            rows = [(self.x, self.buys, self.sells, self.holds, ws, self.pl)]
        
            print(tf.generate_table(rows, cols))
            
        
        return self.state.image, reward, done, {}
        
    
     
    def step(self, action):
        self.actions += 1
        
        if self.continuous:
            if action[0] > 0:
                action = 1
            else:
                action = 0
        
        return self.step_v3(action)
        return self.step_v2(action)
        '''
        
        '''
        self.x += 1
        if (self.x >= len(self.files)):
            self.x = 0


        price_diff = 0
        pl_percent = 0
        
        min_pl = 0
        max_pl = 100
        
        if self.open_trade:
            price_diff = to_pl(self.state.price, self.prev_state.price)
            
            if self.state.time < self.prev_state.time or np.abs(price_diff) > 100:
                self.open_trade = False
                reward = 0
                
            else:    
                if self.trade_type == 1:
                    price_diff = -price_diff
                
                
                if action == 0:
                    self.pl += price_diff - 12.5
                    self.open_trade = False
                    self.closes += 1
                    
                    reward = price_diff - 12.5
                else:
                    self.holds += 1
                   
                   
                    diff_for_trade = to_pl(self.state.price, self.trade_price)
                    if self.trade_type == 1:
                        diff_for_trade = -diff_for_trade
                        
                    pl_percent = np.abs(diff_for_trade) / max_pl
                    
                    if diff_for_trade < 0:
                        pl_percent = -pl_percent
                        
                    reward = price_diff - 12.5
                    
                        
                    #if np.abs(price_diff) > max_pl:
                     #   self.open_trade = False
                      #  self.pl += (price_diff - 18.5)       
                       # self.closes += 1
                       # self.holds -= 1
                       # self.auto_closes += 1
                
        else:
        
            self.trades += 1
            self.open_trade = True
            self.trade_price = self.state.price
            self.trade_type = action
            reward = 0
            if action == 0:
                self.buys += 1
            elif action == 1:
                self.sells += 1
        
        
        
        if self.debug and self.x % 100 == 0:
            cols = ['open_trade','trade_price', 'current_price', 'price_diff', 'pl_percent', 'reward']
            rows = [(self.open_trade, self.trade_price, self.state.price, price_diff, pl_percent, reward)]
            print(tf.generate_table(rows, cols))
            
        
        if pl_percent < -1: pl_percent = -1
        if pl_percent > 1: pl_percent = 1
        
        self.prev_state = self.state
        self.state = SCState(self.dir, self.files[self.x], self.open_trade, action, pl_percent)
        
        if reward > self.high_pl:
            self.high_pl = reward
        
        if reward < self.low_pl:
            self.low_pl = reward
        
        if self.x % 1000 == 0:
            cols = ['buys', 'sells', 'holds', 'high', 'low', 'pl']
            rows = [(self.buys, self.sells, self.holds, self.high_pl, self.low_pl, self.pl)]
        
            print(tf.generate_table(rows, cols))
            
                        
        if self.x % 5000 == 0:
            log_dir = '/home/dan/serpent/frame_log'
            os.makedirs(log_dir, exist_ok=True)
            cv2.imwrite(os.path.join(log_dir, '%d%s' % (self.rank, self.state.file)), self.state.image)
        
        done = False
        if self.trades % 100 == 0:
            done = True
        
        return self.state.image, reward, done, {}
        
        
    def render(self, mode='human', close=False):
        pass
    
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
