from serpent.game import Game

from .api.api import TradingViewSimAPI

from serpent.utilities import Singleton

from serpent.game_launchers.web_browser_game_launcher import WebBrowser

class SerpentTradingViewSimGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "web_browser"

        kwargs["window_name"] = ".*Mozilla.*"

        kwargs["url"] = "https://www.tradingview.com/chart/uBNGW7De/"
        kwargs["browser"] = WebBrowser.DEFAULT

        super().__init__(**kwargs)

        self.api_class = TradingViewSimAPI
        self.api_instance = None

    @property
    def screen_regions(self):
        regions = {
            "SAMPLE_REGION": (0, 0, 0, 0)
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
