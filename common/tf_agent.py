from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts



from serpent.input_controller import KeyboardKey
from serpent.game_agent import GameAgent
# from .ppo import SerpentPPO
from serpent import cv
import tesserocr
from serpent.frame_transformer import FrameTransformer
from PIL import Image
import re
from serpent.input_controller import MouseButton
from serpent.frame_grabber import FrameGrabber
import os
import pickle
# from agent_model import AgentModel
import redis
import json
from datetime import datetime
from redis_helpers import RedisKeys
from serpent import serpent
from serpent.input_controller import InputController
import gym
from gym import spaces
import numpy as np
from serpent.window_controller import WindowController
from serpent.visual_debugger.visual_debugger import VisualDebugger
from time import sleep
import subprocess
import time
# tf.compat.v1.enable_v2_behavior()



class T4TFEnv(py_environment.PyEnvironment):

    def __init__(self, fake=False, metrics_key='001'):
        with open('running', 'w') as f:
            f.write(str(os.getpid()))
        
        self._episode_ended = False

        self.game = serpent.initialize_game('T4TF1')

        game_frame = self.game.screen_regions['GAME_REGION']
        self.width = 10
        self.height = 10

        self.state_shape = (int(self.height / 2), int(self.width / 2), 1)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self.state_shape, dtype=np.float32, minimum=0.0, name='observation')


        self._state = np.zeros(self.state_shape).astype(np.float32)

        if fake:
            return
        self.interrupted = False

        self.game.launch()
        self.game.start_frame_grabber()
        self.input_controller = InputController(game=self.game)
        # self.input_proc = 

        self.frame_buffer = FrameGrabber.get_frames([0])
        self.frame_buffer = self.extract_game_area(self.frame_buffer)

        self.width = self.frame_buffer[0].shape[1]
        self.height = self.frame_buffer[0].shape[0]
        print('width: %d' % self.width)
        print('height: %d' % self.height)
        self.state_shape = (self.height, self.width, 3)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self.state_shape, dtype=np.float32, minimum=0.0, name='observation')

        self._state = np.zeros(self.state_shape).astype(np.float32)

        # print('created input with pid: %s' % self.input_proc.pid)
        self.sell_keys = [KeyboardKey.KEY_LEFT_SHIFT, KeyboardKey.KEY_LEFT_CTRL, KeyboardKey.KEY_S]
        self.buy_keys = [KeyboardKey.KEY_LEFT_SHIFT, KeyboardKey.KEY_LEFT_CTRL, KeyboardKey.KEY_B]
        self.step_keys = [KeyboardKey.KEY_LEFT_SHIFT, KeyboardKey.KEY_LEFT_CTRL, KeyboardKey.KEY_F]


        self.visual_debugger = VisualDebugger()

        self.scraper = T4Scraper(game=self.game, visual_debugger=self.visual_debugger)
        frame = self.game.grab_latest_frame()
        self.scraper.current_frame = frame
        self.pl = 0
        self.working_trade = 0
        self.current_action = ''
        self.held = False
        self.fill_count = 0

        self.window_controller = WindowController()
        self.window_id = self.window_controller.locate_window(".*Mini-Dow .*")
        # self.window_id = self.window_controller.locate_window(".*S&P .*")

        self.keys = RedisKeys(metrics_key)
#         self.redis = redis.Redis(port=6001)
    
        self.number_of_trades = 0
        self.number_of_wins = 0
        self.buys = 0
        self.sells = 0
        self.holds = 0
        self.history = list()
        self.actions = 0
        self.last_action = ''

        self.previous_write = -1
        self.get_metadata()
        
        self.active_frame = None
        
        self.start_time = time.time()
        
        self.step_read_time = 0
        self.step_write_time = 0
        

    def get_state(self, zeros=False):
        if zeros:
            return np.zeros(self.state_shape).astype(np.float32)
            self.frame_history = [
                np.zeros((int(self.width / 4), int(self.height / 4))).astype(np.float32),
                np.zeros((int(self.width / 4), int(self.height / 4))).astype(np.float32),
                np.zeros((int(self.width / 4), int(self.height / 4))).astype(np.float32),
                np.zeros((int(self.width / 4), int(self.height / 4))).astype(np.float32)
            ]
            
        st = np.stack(
            self.frame_history,
            axis=2
        ).astype(np.float32)
        
        print(st.shape)
        return st

    
    def is_focused(self):
        return self.window_controller.is_window_focused(self.window_id)

    def display_frame(self):
        frame = self.game.grab_latest_frame()
        self.visual_debugger.store_image_data(
                frame.frame,
                frame.frame.shape,
                2
            )

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        print('RESTARTING')
        self._state = np.zeros(self.state_shape).astype(np.float32)
        self._episode_ended = False
        return ts.restart(self._state)


    def update_scraper_frame(self):
        self.scraper.current_frame = self.game.grab_latest_frame()
        self.visual_debugger.store_image_data(
            self.scraper.current_frame.frame,
            self.scraper.current_frame.frame.shape,
            str(2)
        )

    def stop(self):
        self.input_proc.kill()
        self.game.stop_frame_grabber()
        return ts.termination(self._state, 0)
    
    def write_order(self, order_type):
        write_start = time.time()
#         if order_type is self.previous_write: return
        with open('/home/dan/.wine/drive_c/input.txt', 'w') as f:
            f.write('%d' % (order_type))
        
        self.step_write_time += (time.time() - write_start)
        self.previous_write = order_type

    def step_forward(self):
        self.write_order(3)
        
    def add_to_history(self, frame, action, reward):
        history_start = time.time()
        im = Image.fromarray(frame)
        im.save('history/%d_%d_%f.jpg' % (int(datetime.now().timestamp() * 100), action, reward))
        print("history add time: %s seconds" % (time.time() - history_start))
#     def log(self, string):
#         if self.actions % 200 == 0:
#             print(string)
#      
    def _step(self, action):
        self.step_read_time = 0
        self.step_write_time = 0
        
        if self.interrupted:
            return self.stop()

        print('----')
        if self._episode_ended:
        # The last action ended the episode. Ignore the current action and start
        # a new episode.
            return self.reset()

        if action == 0:
            # perform buy
            self.current_action = 'buy'
            self.working_trade = True
            # self.input_controller.tap_keys(self.buy_keys, duration=0.001)
            self.write_order(action)

        elif action == 1:
            # perform sell
            self.current_action = 'sell'
            self.working_trade = True
            # self.input_controller.tap_keys(self.sell_keys, duration=0.001)
            self.write_order(action)
        elif action == 2:
            print('hold')
            self.current_action = 'hold'
            y = 0 
            while y < 3:
#                 self.write_order(action)
                self.step_forward()
                sleep(0.1)
                y = y + 1
                
        self.actions += 1

        if action < 2:
            self.number_of_trades += 1
        
            while not self.has_open_positions():
                sleep(0.01)

                if self.interrupted:
                    return self.stop()


            while self.has_open_positions():
                self.step_forward()
                sleep(0.01)

                if self.interrupted:
                    return self.stop()
        
        
        self.write_order(4)
        
        
        sleep(0.2)
        reward = self.reward_agent()

        if self.actions > 1:
            self.add_to_history(self.active_frame, action, reward)

        start_grab_frame = time.time()
        self.frame_buffer = FrameGrabber.get_frames([0])
        self.frame_buffer = self.extract_game_area(self.frame_buffer)
        print("frame grab time: %s seconds" % (time.time() - start_grab_frame))
        self.active_frame = self.frame_buffer[0]
#         for i, game_frame in enumerate(self.frame_buffer):
#             if i >= 3: break
#             self.visual_debugger.store_image_data(
#                 game_frame,
#                 game_frame.shape,
#                 str(i)
#             )
        print(self.frame_buffer[0].shape)    

#         self.frame_history.insert(0, self.frame_buffer[0])
        
#         self._states = self.get_state()
        states_shape = self.state_shape
        print(states_shape)
        self._states = np.reshape(self.frame_buffer[0], (states_shape[0], states_shape[1], states_shape[2])).astype(np.float32)
        print(self._states.shape)
        if self.number_of_trades > 0:
            self.push_metadata(action=self.current_action, reward=reward)
        
            print('Wins: %d%% - %d / %d' % ((self.number_of_wins / self.number_of_trades * 100), self.number_of_wins, self.number_of_trades))
            
        
        # print history
        
        if len(self.history) > 0:
            historical_wins = len(list(filter(lambda x: x[1] > 0, self.history)))
            print('Wins (last %d): %d%%' % (len(self.history), (historical_wins / len(self.history) * 100)))
        
        print('Buys: %d' % self.buys)
        print('Sells: %d' % self.sells)
        print('Holds: %d' % self.holds)
        
        print('Step read time: %s' % self.step_read_time)
        print('Step write time: %s' % self.step_write_time)
        # print(states.shape)
        return ts.transition(self._states, reward=reward, discount=1.0)

    def read_position_and_pl(self):
        read_start = time.time()
        result = ['','']
        while len(result[0]) < 1 or len(result[1]) < 1:
            with open('/home/dan/.wine/drive_c/output.txt', 'r') as f:
                result = [x.strip() for x in f.read().split(',')]
        
        self.step_read_time += (time.time() - read_start)
        return (int(result[0]), int(result[1]))

    def has_open_positions(self):
        result = self.read_position_and_pl()
        pos = result[0]
        pl = result[1]
        
        # if self.working_trade:
        #     if pos != 0:
        #         self.working_trade = False
        #     else:
        #         if pl != self.pl:
        #             self.working_trade = False
        #             return False
        #         else:
        #             return True
        
        if pos != 0:
            return True
        return False

    def reward_agent(self):
        # get pl for last trade

        newPL = self.read_position_and_pl()[1]
        print('old pl: %d' % self.pl)
        print('new pl: %d' % newPL)
        if newPL > self.pl:
            reward = 1.0
        else:
            reward = -1.0

        if self.current_action is 'hold':
            reward = -0.25
        else:
            if reward > 0:
                if self.last_action is 'hold':
                    reward = 1.25
        # if reward is 1.0:
        #     if self.current_action is 'buy' and self.buys > self.sells:
        #         reward = 0.8
        #     elif self.current_action is 'sell' and self.sells > self.buys:
        #         reward = 0.8
        # elif reward is -1.0:
        #     if self.current_action is 'sell' and self.buys > self.sells:
        #         reward = -0.8
        #     elif self.current_action is 'buy' and self.sells > self.buys:
        #         reward = -0.8
                
        
        self.last_action = self.current_action
        print('REWARD: %f' % reward)
        self.pl = newPL
        return reward
        
    def extract_game_area(self, frame_buffer):
        game_area_buffer = []
#         for game_frame in frame_buffer.frames:
#             game_area = cv.extract_region_from_image(
#                 game_frame.grayscale_frame,
#                 self.game.screen_regions["GAME_REGION"]
#             )

        frame = frame_buffer.frames[0].quarter_resolution_frame
#         frame = FrameTransformer.rescale(frame_buffer.frames[0].grayscale_frame, 0.5)
        game_area_buffer.append(frame)

        return game_area_buffer


    def get_metadata(self):
        return
        if self.redis.exists(self.keys.trades):
            self.number_of_trades = self.redis.llen(self.keys.trades)
            
            self.history = list()
            history_strings = self.redis.lrange(self.keys.trades, -100, 100)
            for s in history_strings:
                d = json.loads(s)
                self.history.append((d['action'], float(d['reward'])))
        
        if self.redis.exists(self.keys.wins):
            self.number_of_wins = int(self.redis.get(self.keys.wins))
            
        if self.redis.exists(self.keys.buys):
            self.buys = int(self.redis.get(self.keys.buys))
        
        if self.redis.exists(self.keys.sells):
            self.sells = int(self.redis.get(self.keys.sells))
        
        if self.redis.exists(self.keys.holds):
            self.holds = int(self.redis.get(self.keys.holds))

    def push_metadata(self, action, reward, reset=False):
        return
        if action is None:
            return
            
        # last 100
        if not action is 'hold':
            while len(self.history) >= 100:
                self.history.pop(0)    
    
            self.history.append((action, reward))

        
        obj = {
            'timestamp':str(datetime.now().timestamp()),
            'action': action,
            'reward': str(reward)
        }
        
        self.redis.rpush(self.keys.trades, json.dumps(obj))
        
        if action == 'buy':
            self.buys += 1
            self.redis.incr(self.keys.buys)
        elif action == 'sell':
            self.sells += 1
            self.redis.incr(self.keys.sells)
        elif action == 'hold':
            self.holds += 1
            self.redis.incr(self.keys.holds)
            
        if reward > 0:
            self.number_of_wins += 1
            self.redis.incr(self.keys.wins)
            
        if reset:
            self.redis.set(self.keys.buys, "0")
            self.redis.set(self.keys.sells, "0")
            self.redis.set(self.keys.wins, "0")
            self.redis.set(self.keys.trades, "0")
            self.redis.set(self.keys.holds, "0")

class T4Scraper:

    def __init__(self, game, visual_debugger):
        self.game = game
        self.current_frame = None
        self.visual_debugger = visual_debugger

    def get_text(self, region, game_frame):
        area = cv.extract_region_from_image(
            game_frame.grayscale_frame,
            self.game.screen_regions[region]
        )

        return tesserocr.image_to_text(Image.fromarray(area))

    def get_market_change(self):
        area = cv.extract_region_from_image(
            self.current_frame.grayscale_frame,
            self.game.screen_regions["MARKET_CHANGE"]
        )

        text = tesserocr.image_to_text(Image.fromarray(area))
        matches = re.findall("\d*", text)
        for match in matches:
            if len(match) > 0:
                return int(match.strip())

        return -1

    def get_working_buys(self):
        area = cv.extract_region_from_image(
            self.current_frame.grayscale_frame,
            self.game.screen_regions["WORKING_BUYS"]
        )

        self.visual_debugger.store_image_data(
            area,
            area.shape,
            3
        )

        text = tesserocr.image_to_text(Image.fromarray(area))
        matches = re.findall("\(\d\)", text)
        for match in matches:
            if len(match) > 0:
                return int(match.strip(' ()'))

        return -1

    def get_working_sells(self):
        area = cv.extract_region_from_image(
            self.current_frame.grayscale_frame,
            self.game.screen_regions["WORKING_SELLS"]
        )

        self.visual_debugger.store_image_data(
            area,
            area.shape,
            2
        )

        text = tesserocr.image_to_text(Image.fromarray(area))
        matches = re.findall("\(\d\)", text)
        for match in matches:
            if len(match) > 0:
                return int(match.strip(' ()'))

        return -1

    def get_position_count(self):
        area = cv.extract_region_from_image(
            self.current_frame.grayscale_frame,
            self.game.screen_regions["POSITIONS"]
        )
        
        self.visual_debugger.store_image_data(
            area,
            area.shape,
            4
        )


        text = tesserocr.image_to_text(Image.fromarray(area))
        # print(text)
        if "Flat" in text:
            return 0

        matches = re.findall("\d", text)
        # print(matches)

        position = 0
        if len(matches) > 0:
            for match in matches:
                stripped = match.strip()
                if len(stripped) > 0:
                    position = int(stripped)
                    break

        # print('position: %d' % position)
        return position

    def get_position_and_fill_count(self):
        area = cv.extract_region_from_image(
            self.current_frame.grayscale_frame,
            self.game.screen_regions["POSITIONS"]
        )
        
        self.visual_debugger.store_image_data(
            area,
            area.shape,
            4
        )


        text = tesserocr.image_to_text(Image.fromarray(area))

        matches = re.findall("\d*\s", text)
#         print(matches)

        position = 0
        if len(matches) > 0:
            for match in matches:
                stripped = match.strip()
                if len(stripped) > 0:
                    position = int(stripped)

        fills = 0
        matches = re.findall("\(.*-", text)
        if len(matches) > 0:
            for match in matches:
                if len(match) > 0:
                    numbers = re.findall("\d*", match)
                    for number in numbers:
                        if len(number) > 0:
                            fills = int(number)

        return (position, fills)

    def get_pl(self):

        area = cv.extract_region_from_image(
            self.current_frame.grayscale_frame,
            self.game.screen_regions["PL"]
        )

        self.visual_debugger.store_image_data(
            area,
            area.shape,
            3
        )

        text = tesserocr.image_to_text(Image.fromarray(area), psm=tesserocr.PSM.SINGLE_LINE)
#         print(text)
        
        
        matches = re.findall("-?\d*", text)
#         print(matches)
        if len(matches) > 0:
            c = ''
            for match in matches:
                stripped = match.strip()
                c = c + stripped

            if len(c) > 0:
                c = re.sub("(?<=\d)\D",'',c)
                return int(c)
        print('DID NOT FIND PL')

        return 0