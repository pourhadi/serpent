import sys
sys.path.append('/home/dan/serpent/common')


from serpent.game_agent import GameAgent
from .helpers.ppo import SerpentPPO
import serpent.cv
import tesserocr
from serpent.frame_transformer import FrameTransformer
from PIL import Image
import re
from serpent.input_controller import MouseButton
from serpent.frame_grabber import FrameGrabber
import os
# from .helpers.agent import Agent

from agent import Agent

class SerpentT4v2GameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handlers["TEST"] = self.handle_test

        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_setups["TEST"] = self.setup_test
        
        self.agent = Agent(self)

    def setup_test(self):
        self.scraper = T4Scraper(self.game)
        self.frame_buffer = None

    def handle_test(self, game_frame):
        self.scraper.current_frame = game_frame
        position = self.scraper.get_position()
        pl = self.scraper.get_pl()
        print(position)
        print(pl)
        print('-----')

    def setup_play(self):
        game_inputs = {
            "Buy": 1,
            "Sell": 2,
        }
		
        metadata = "v2"
        model = "t4v2model"
        ppo_agent = SerpentPPO(
            frame_shape=(269, 252, 4),
            game_inputs=game_inputs,
            model_name=model
        )
        self.agent.setup(ppo_agent, model_name=model, metadata_key=metadata)

    def handle_play(self, game_frame):
        self.agent.play(game_frame)
    
        
    
    def pull_trades(self):
        self.input_controller.move(x=self.pull_point[0], y=self.pull_point[1])
        self.input_controller.click()

