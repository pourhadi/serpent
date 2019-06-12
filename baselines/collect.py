
from serpent.input_controller import KeyboardKey
from serpent.game_agent import GameAgent
# from .ppo import SerpentPPO
from serpent import cv
import tesserocr
from serpent.frame_transformer import FrameTransformer
from PIL import Image
import re
from serpent.input_controller import MouseButton
from serpent.frame_grabber import FrameGrabber
import os
import pickle
# from agent_model import AgentModel
import redis
import json
from datetime import datetime
from serpent import serpent
from serpent.input_controller import InputController
import gym
from gym import spaces
import numpy as np
from serpent.window_controller import WindowController
from serpent.visual_debugger.visual_debugger import VisualDebugger
from time import sleep
import subprocess
import time

game = serpent.initialize_game('T4TF1')
game.launch()
game.start_frame_grabber()

def add_to_history(frame, name):
    im = Image.fromarray(frame)
    im.save('frames_dow/%s.jpg' % name)
    print(name)
    
def write_order(order_type):
    with open('/home/dan/.wine/drive_c/input.txt', 'w') as f:
        f.write('%d' % (order_type))
    
def read_state():
    result = ['','']
    while len(result[0]) < 1 or len(result[1]) < 1:
        with open('/home/dan/.wine/drive_c/output.txt', 'r') as f:
            result = [x.strip() for x in f.read().split(',')]
    
    return (int(result[0]), int(result[1]), float(result[2]), int(result[3]))

last_name = ''
command = 7
while True:
    
    _, _, price, time = read_state()
    
    frame = FrameGrabber.get_frames([0]).frames[0].half_resolution_frame
    
    name = '%d_%f' % (time, price)
    
    if name != last_name:
        last_name = name
        add_to_history(frame, name)
    
    if command == 6:
        command = 7
    else:
        command = 6
    
    write_order(command)
    
    sleep(0.1)