import sys
sys.path.append('/home/dan/serpent/common')

from tf_agent import T4TFEnv

import gym
import numpy as np

from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines import ACER


import gym
from gym import spaces
from t4_env import T4Env

env = DummyVecEnv([lambda: T4Env()])

# print(env.action_space.high)
# print(env.action_space.low)
# env = gym.make('Breakout-v0')
# env = DummyVecEnv([lambda: env])

model = PPO2('CnnPolicy', env, verbose=1, tensorboard_log="./tb_log/")
# model = ACER.load("acer_t4")
# model.set_env(env)
model.learn(total_timesteps=5000, log_interval=10)
model.save("ppo_t4")

# del model # remove to demonstrate saving and loading



# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
