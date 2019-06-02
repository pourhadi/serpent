from baselines.t4_env import T4Env
import os
import cv2

class PretrainEnv(T4Env):
    
    def set_files(self, files, dir):
        self.files = files
        self.x = 0
        self.dir = dir
        self.x_reward = 0.0
        self.x_action = 'hold'
        
    def step(self, action):
        if self.x >= len(self.files):
            return [], self.x_reward, True, {}
        
        file = self.files[self.x]
        im = cv2.imread(os.path.join(self.dir, file))
        
        ret_action = self.x_action
        ret_reward = self.x_reward
        obs = im
        
        split = file.split('_')
        self.x_reward = float(split[2].strip('.jpg'))
        
        if split[1] == 'buy':
            action = 0
        elif split[1] == 'sell':
            action = 1
        else: action = 2
        
        self.x_action = action
        
        self.x = self.x + 1
        return obs, ret_reward, False, {}

dir = '/home/dan/serpent/history_processed/'
files = os.listdir(dir)

env = PretrainEnv()
env.set_files(files, dir)

def model_fn(obs):
    return env.x_action
    
from stable_baselines.gail import generate_expert_traj

generate_expert_traj(model_fn, 't4_dataset', env, n_episodes=1, image_folder='recorded_images')