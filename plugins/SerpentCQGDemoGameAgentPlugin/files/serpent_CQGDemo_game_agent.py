from serpent.game_agent import GameAgent
from .helpers.ppo import SerpentPPO

class SerpentCQGDemoGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):
        
        self.game_data = CQGScraper(self.game)
        self.frame_buffer = None
        
        game_inputs = {
            "Buy": 0,
            "Sell": 1,
            "Hold": 2
        }
        
        self.ppo_agent = SerpentPPO(
            frame_shape=(512, 344, 4),
            game_inputs=game_inputs
        )
        
        try:
            self.ppo_agent.agent.restore_model(directory=os.path.join(os.getcwd(), "datasets", "cqgdemoai"))
        except Exception:
            pass
            
        game_frame_buffer = FrameGrabber.get_frames([0, 1, 2, 3], frame_type="PIPELINE")
        game_frame_buffer = self.extract_game_area(game_frame_buffer)   
        self.ppo_agent.generate_action(game_frame_buffer)
        
        self.pl = 0
        
    def handle_play(self, game_frame):
        self.game_data.current_frame = game_frame
        
        if self.has_open_positions():
            return
        
        reward = self.reward_agent()
        
        if self.frame_buffer is not None:
            if self.ppo_agent.agent.batch_count == self.ppo_agent.agent.batch_size - 1:
                self.ppo_agent.observe(reward, terminal=False)
                self.frame_buffer = None
            else:
                self.ppo_agent.observe(reward, terminal=False)
        
        self.frame_buffer = FrameGrabber.get_frames([0,1,2,3], frame_type="PIPELINE")
        self.frame_buffer = self.extract_game_area(self.frame_buffer)
        
        action, label, game_input = self.ppo_agent.generate_action(self.frame_buffer)
        
        if game_input == 0:
            # perform buy
            pass
        elif game_input == 1:
            # perform sell
            pass
        else:
            # hold
            pass
            
    def reward_agent(self):
        # get pl for last trade
        pass
        
    def has_open_positions(self):
        # return whether there's an open position
        return False
        
class CQGScraper:
    def __init__(self, game):
        self.current_frame = None
        self.game = game
        
        
