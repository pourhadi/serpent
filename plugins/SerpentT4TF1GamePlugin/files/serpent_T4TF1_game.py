from serpent.game import Game

from .api.api import T4TF1API

from serpent.utilities import Singleton




class SerpentT4TF1Game(Game, metaclass=Singleton):

	
    def __init__(self, **kwargs):
        kwargs["platform"] = "executable"

        kwargs["window_name"] = ".*Mini-Dow .*"
        # kwargs["window_name"] = ".*S&P .*"

        

        kwargs["executable_path"] = "wine"
        

        super().__init__(**kwargs)

        self.api_class = T4TF1API
        self.api_instance = None

    def stop(self):
        super().stop_input_controller()

    @property
    def screen_regions(self):
        regions = {
            "USERNAME": (137, 53, 163, 210),
            "POSITIONS": (118, 16, 139, 121),
            "PL": (166, 16, 186, 133),
            "GAME_REGION": (20, 7, 782, 1020),
            #"GAME_REGION": (21, 234, 582, 1014),
            "WORKING_BUYS": (130, 50, 153, 84),
            "WORKING_SELLS": (130, 169, 154, 220),
            "MARKET_CHANGE": (168, 153, 188, 234)
        }

        return regions

    @property
    def ocr_presets(self):
        presets = {
            "SAMPLE_PRESET": {
                "extract": {
                    "gradient_size": 1,
                    "closing_size": 1
                },
                "perform": {
                    "scale": 10,
                    "order": 1,
                    "horizontal_closing": 1,
                    "vertical_closing": 1
                }
            }
        }

        return presets
