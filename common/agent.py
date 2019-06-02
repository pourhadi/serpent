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

class Agent:

    def screen_regions():
        regions = {
            "USERNAME": (137, 53, 163, 210),
            "POSITIONS": (84, 286, 121, 556),
            "PL": (153, 296, 188, 495),
            "GAME_REGION": (284, 2, 822, 506),
            "WORKING_BUYS": (130, 50, 153, 84),
            "WORKING_SELLS": (130, 169, 154, 220),
            "MARKET_CHANGE": (168, 153, 188, 234)
        }
        return regions


    def __init__(self, s):
        self.s = s

    def setup(self, ppo_agent, model_name, metadata_key):
    
        self.keys = RedisKeys(metadata_key)
        self.redis = redis.Redis(port=6001)
    
        self.ppo_agent = ppo_agent
        self.history = list()
        self.m_k = metadata_key
        self.buys = 0
        self.sells = 0
        self.last_frame_had_position = False
        self.fill_count = 0
        self.working_trade = False
        self.episode_count = 0

        self.number_of_trades = 0
        self.number_of_wins = 0
        self.current_action = None
        self.last_action = None
        self.repeat_count = 0

        self.sell_point = (671, 447)
        self.buy_point = (669, 476)
        self.pull_point = (562, 173)
        self.held = False
        self.scraper = T4Scraper(self.s.game, self.s.visual_debugger)
        self.frame_buffer = None

        self.scraper.current_frame = FrameGrabber.get_frames([0]).frames[0]
        self.pl = self.scraper.get_pl()
        self.fill_count = self.scraper.get_position_and_fill_count()[1]
        
        self.model_name = model_name

        
        print('AFTER INIT AGENT')
        try:
            self.ppo_agent.agent.restore(directory=os.path.join(
                os.getcwd(), "datasets", self.model_name))
#             self.ppo_agent.agent.restore(directory=os.path.join(os.getcwd(), "datasets", "t4simmodel"))
        except Exception:
            pass
            
        print('AFTER RESTORE')
        
        self.get_metadata()
        
#         try:
#             with open(self.metadata_filename, 'rb') as f:
#                 print('load metadata')
#                 data = pickle.loads(f.read())

#             self.number_of_trades = data['trades']
#             self.number_of_wins = data['wins']
#             self.buys = data['buys']
#             self.sells = data['sells']
#         except FileNotFoundError:
#             data = {'trades':0, 'wins':0, 'buys':0, 'sells':0}
#             with open(self.metadata_filename, 'wb') as f:
#                 print('create metadata file')
#                 f.write(pickle.dumps(data))
                
    def get_metadata(self):
        if self.redis.exists(self.keys.trades):
            self.number_of_trades = self.redis.llen(self.keys.trades)
            
            self.history = list()
            history_strings = self.redis.lrange(self.keys.trades, -100, 100)
            for s in history_strings:
                d = json.loads(s)
                self.history.append((d['action'], int(d['reward'])))
        
        if self.redis.exists(self.keys.wins):
            self.number_of_wins = int(self.redis.get(self.keys.wins))
            
        if self.redis.exists(self.keys.buys):
            self.buys = int(self.redis.get(self.keys.buys))
        
        if self.redis.exists(self.keys.sells):
            self.sells = int(self.redis.get(self.keys.sells))

    def push_metadata(self, action, reward):
        if action is None:
            return
            
        # last 100
        while len(self.history) >= 100:
            self.history.pop(0)    

        self.history.append((action, reward))

        
        obj = {
            'timestamp':str(datetime.now().timestamp()),
            'action': action,
            'reward': str(reward)
        }
        
        self.redis.rpush(self.keys.trades, json.dumps(obj))
        
        if action == 'Buy':
            self.buys += 1
            self.redis.incr(self.keys.buys)
        elif action == 'Sell':
            self.sells += 1
            self.redis.incr(self.keys.sells)
            
        if reward > 0:
            self.number_of_wins += 1
            self.redis.incr(self.keys.wins)
        

    def play(self, game_frame):
        self.scraper.current_frame = game_frame
        
        if not self.held and self.has_open_positions():
            self.last_frame_had_position = True
            return

        if self.last_frame_had_position:
            print('last frame had open position')
            self.last_frame_had_position = False
            return

        self.episode_count += 1
        reward = self.reward_agent()
        
        
        if self.number_of_trades > 0:
            self.push_metadata(action=self.current_action, reward=reward)
        
            print('Wins: %d%% - %d / %d' % ((self.number_of_wins / self.number_of_trades * 100), self.number_of_wins, self.number_of_trades))
            
        
        # print history
        
        if len(self.history) > 0:
            historical_wins = len(list(filter(lambda x: x[1] > 0, self.history)))
            print('Wins (last %d): %d%%' % (len(self.history), (historical_wins / len(self.history) * 100)))
        
        
        
        print('Buys: %d' % self.buys)
        print('Sells: %d' % self.sells)
       
        
#         if self.number_of_trades % 10 == 0:
#             data = {'trades': self.number_of_trades, 'wins' : self.number_of_wins, 'buys':self.buys, 'sells':self.sells}
#             with open(self.metadata_filename, 'wb') as f:
#                 f.write(pickle.dumps(data))
        
        self.ppo_agent.observe(reward, terminal=False)

        self.frame_buffer = game_frame
        self.frame_buffer = FrameGrabber.get_frames([0, 1, 2, 3], frame_type="PIPELINE")
        self.frame_buffer = self.extract_game_area(self.frame_buffer)

        
        for i, game_frame in enumerate(self.frame_buffer):
            if i >= 3: break
            self.s.visual_debugger.store_image_data(
                game_frame,
                game_frame.shape,
                str(i)
            )

        
        action, label, game_input = self.ppo_agent.generate_action(self.frame_buffer)
        print(label)
        self.current_action = label
        
        self.number_of_trades += 1
    
        if game_input == 1:
            # perform buy
            
            self.working_trade = True
            self.s.input_controller.move(
                x=self.buy_point[0], y=self.buy_point[1])
            self.s.input_controller.click()

        elif game_input == 2:
            # perform sell
            
            self.working_trade = True
            self.s.input_controller.move(
                x=self.sell_point[0], y=self.sell_point[1])
            self.s.input_controller.click()

        elif game_input == 3:
            self.held = True
            
    def reward_agent(self):
        # get pl for last trade

        newPL = int(self.scraper.get_pl())
        print('old pl: %d' % self.pl)
        print('new pl: %d' % newPL)
        print('---')
        if newPL > self.pl:
            reward = 1
        else:
            reward = -1

        if self.held:
            reward = 0.25
            if self.last_action == self.current_action:
                reward = 0
            self.held = False

        self.last_action = self.current_action
        print('REWARD: %d' % reward)
        self.pl = newPL
        return reward
        
    def has_open_positions(self):
        pos, fills = self.scraper.get_position_and_fill_count()
        if self.working_trade:
            if pos == 0 and fills == self.fill_count:
                return True
            self.working_trade = False

        self.fill_count = fills
        if pos > 0:
            return True
        return False

    def extract_game_area(self, frame_buffer):
        game_area_buffer = []
        for game_frame in frame_buffer.frames:
            game_area = serpent.cv.extract_region_from_image(
                game_frame.grayscale_frame,
                self.s.game.screen_regions["GAME_REGION"]
            )

            frame = FrameTransformer.rescale(game_area, 0.5)
            game_area_buffer.append(frame)

        return game_area_buffer

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
