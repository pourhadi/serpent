from serpent.input_controller import InputController
import sys
from serpent.input_controller import KeyboardKey
from time import sleep


step_keys = [KeyboardKey.KEY_LEFT_SHIFT, KeyboardKey.KEY_LEFT_CTRL, KeyboardKey.KEY_F]
input_controller = InputController(game=None)


command = sys.argv[1]
if command is 'f':
    input_controller.press_keys(step_keys)
    sleep(0.001)
    input_controller.release_keys(step_keys)
