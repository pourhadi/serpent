from serpent.game_agent import GameAgent
# from .ppo import SerpentPPO
import serpent.cv
import tesserocr
from serpent.frame_transformer import FrameTransformer
from PIL import Image
import re
from serpent.input_controller import MouseButton
from serpent.frame_grabber import FrameGrabber
import os
import pickle
from agent_model import AgentModel
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

class T4Env(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(T4Env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2)
        # Example for using image as input:

        self.observation_space = spaces.Box(low=0, high=255, shape=
                        (0, 0, 0), dtype=np.uint8)

        self.game = serpent.initialize_game('T4v2')
        self.game.launch()
        self.input_controller = InputController(game=self.game)

        self.sell_point = (671, 447)
        self.buy_point = (669, 476)

        self.visual_debugger = VisualDebugger()
        self.scraper = T4Scraper(game=self.game, visual_debugger=self.visual_debugger)


    def step(self, action):
        pass
        # Execute one time step within the environment
    
    def reset(self):
        pass
        # Reset the state of the environment to an initial state
    
    def render(self, mode='human', close=False):
        pass
        # Render the environment to the screen
    
    def click(self):
        self.game.launch()
        self.input_controller.move(
                x=self.buy_point[0], y=self.buy_point[1])
        self.input_controller.click()




class T4Scraper:

    def __init__(self, game, visual_debugger):
        self.game = game
        self.current_frame = None
        self.visual_debugger = visual_debugger

    def get_text(self, region, game_frame):
        area = serpent.cv.extract_region_from_image(
            game_frame.grayscale_frame,
            self.game.screen_regions[region]
        )

        return tesserocr.image_to_text(Image.fromarray(area))

    def get_market_change(self):
        area = serpent.cv.extract_region_from_image(
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
        area = serpent.cv.extract_region_from_image(
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
        area = serpent.cv.extract_region_from_image(
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

    def get_position_and_fill_count(self):
        area = serpent.cv.extract_region_from_image(
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

        area = serpent.cv.extract_region_from_image(
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
                return int(c)
        print('DID NOT FIND PL')

        return 0