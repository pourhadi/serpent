from serpent.game import Game

from .api.api import T4AndroidAPI

from serpent.utilities import Singleton




class SerpentT4AndroidGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "executable"

        kwargs["window_name"] = "T4 Mobile"

        
        
        kwargs["executable_path"] = "anbox launch --package=com.t4login.t4android --component=com.t4login.t4android.MainActivity"
        
        

        super().__init__(**kwargs)

        self.api_class = T4AndroidAPI
        self.api_instance = None

    @property
    def screen_regions(self):
        regions = {
            "USERNAME": (137, 53, 163, 210),
            "POSITIONS": (193, 3, 219, 179),
            "PL": (95, 3, 120, 141),
            "GAME_REGION": (218, 2, 714, 1022)
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
