from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey

class SerpentTradingViewSimGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):
        inputs = {
            "Buy": [KeyboardKey.KEY_LEFT_SHIFT, KeyboardKey.KEY_B],
            "Sell": [KeyboardKey.KEY_LEFT_SHIFT, KeyboardKey.KEY_S],            
        }
        
    def handle_play(self, game_frame):
        pass
