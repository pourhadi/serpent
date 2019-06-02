from serpent.game import Game

from .api.api import T4v3API

from serpent.utilities import Singleton




class SerpentT4v3Game(Game, metaclass=Singleton):

	
    def __init__(self, **kwargs):
        kwargs["platform"] = "executable"

        kwargs["window_name"] = ".*Windows .*"

        

        kwargs["executable_path"] = "vboxmanage startvm Windows"
        
        

        super().__init__(**kwargs)

        self.api_class = T4v3API
        self.api_instance = None

    @property
    def screen_regions(self):
        regions = {
            "USERNAME": (137, 53, 163, 210),
            "POSITIONS": (84, 232, 121, 556),
            "PL": (153, 232, 188, 495),
            "GAME_REGION": (284, 2, 822, 506),
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
