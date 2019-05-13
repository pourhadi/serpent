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

class SerpentT4AndroidGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handlers["TEST"] = self.handle_test
        
        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_setups["TEST"] = self.setup_test

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
        
        self.fill_count = 0
        self.working_trade = False
        self.episode_count = 0
        
        self.buy_point = (326, 334)
        self.sell_point = (647, 634)
    
        self.scraper = T4Scraper(self.game, self.visual_debugger)
        self.frame_buffer = None
        
        self.scraper.current_frame = FrameGrabber.get_frames([0]).frames[0]
        self.pl = self.scraper.get_pl()
        self.fill_count = self.scraper.get_position_and_fill_count()[1]
        game_inputs = {
            "Buy": 1,
            "Sell": 2
        }
        
        self.ppo_agent = SerpentPPO(
            frame_shape=(248, 510, 2),
            game_inputs=game_inputs
        )
        
        try:
            self.ppo_agent.agent.restore(directory=os.path.join(os.getcwd(), "datasets", "t4androidmodel"))
        except Exception:
            pass
#             
#         game_frame_buffer = FrameGrabber.get_frames([0, 1, 2, 3], frame_type="PIPELINE")
#         game_frame_buffer = self.extract_game_area(game_frame_buffer)   
#         self.ppo_agent.generate_action(game_frame_buffer)
#        
        
    def handle_play(self, game_frame):
        self.visual_debugger.store_image_data(
            game_frame.frame,
            game_frame.frame.shape,
            2
        )
        
#         for i, game_frame in enumerate(self.game_frame_buffer.frames):
#             self.visual_debugger.store_image_data(
#                 game_frame.frame,
#                 game_frame.frame.shape,
#                 str(i)
#             )
#         
        
        self.scraper.current_frame = game_frame
        if self.has_open_positions():
            return
        
        self.episode_count += 1
        reward = self.reward_agent()
        
        self.ppo_agent.observe(reward, terminal=False)

        self.frame_buffer = game_frame        
        self.frame_buffer = FrameGrabber.get_frames([0,4], frame_type="PIPELINE")
        self.frame_buffer = self.extract_game_area(self.frame_buffer)
        
        self.visual_debugger.store_image_data(
            self.frame_buffer[0],
            self.frame_buffer[0].shape,
            3
        )
        action, label, game_input = self.ppo_agent.generate_action(self.frame_buffer)
        print(label)
        
        if game_input == 1:
            # perform buy
            self.working_trade = True
            self.input_controller.move(x=self.buy_point[0], y=self.buy_point[1])
            self.input_controller.click()
        
        elif game_input == 2:
            # perform sell
            self.working_trade = True
            self.input_controller.move(x=self.sell_point[0], y=self.sell_point[1])
            self.input_controller.click()
#         
#         if self.episode_count % 5 == 0:
#             print('start save')
# #             self.ppo_agent.agent.save(directory=os.path.join(os.getcwd(), "datasets", "t4androidmodel"), append_timestep=False)
#             print('save complete')
            
    def reward_agent(self):
        # get pl for last trade
        
        newPL = int(self.scraper.get_pl())
        print('old pl: %d' % self.pl)
        print('new pl: %d' % newPL)
        print('---')
        if newPL > self.pl:
            reward = 1
        else: reward = -2
        
        print('REWARD: %d' % reward)
        self.pl = newPL
        return reward

    def has_open_positions(self):
        pos, fills = self.scraper.get_position_and_fill_count()
        if self.working_trade:
            if pos == 0 and fills == self.fill_count: return True
            self.working_trade = False
        
        self.fill_count = fills
        if pos > 0: return True
        return False
    
    def extract_game_area(self, frame_buffer):
        game_area_buffer = []
        for game_frame in frame_buffer.frames:
            game_area = serpent.cv.extract_region_from_image(
                game_frame.grayscale_frame,
                self.game.screen_regions["GAME_REGION"]
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

    
    def get_position_and_fill_count(self):
        area = serpent.cv.extract_region_from_image(
            self.current_frame.grayscale_frame,
            self.game.screen_regions["POSITIONS"]
        )
        
        
        self.visual_debugger.store_image_data(
            area,
            area.shape,
            0
        )
        
        text = tesserocr.image_to_text(Image.fromarray(area))
#         print(text)
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
            1
        )
        
        text = tesserocr.image_to_text(Image.fromarray(area))
        print(text)
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