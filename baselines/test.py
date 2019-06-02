import gym
import numpy as np

from stable_baselines.sac.policies import LnCnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC


import gym
from gym import spaces

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    
    self._env = gym.make('Pong-v0')

    self.action_space = spaces.Box(low=-(self._env.action_space.n / 2), high=(self._env.action_space.n / 2)-1, shape=(1,), dtype=np.int8)
    # Example for using image as input:
    self.observation_space = self._env.observation_space
  def step(self, action):
    # print('action')  
    # print((self._env.action_space.n / 2) + action[0])
    return self._env.step(int((self._env.action_space.n / 2) + action[0]))
    
  def reset(self):
    return self._env.reset()

  def render(self, mode='human', close=False):
    return self._env.render(mode=mode, close=close)


env = DummyVecEnv([lambda: CustomEnv()])

print(env.action_space.high)
print(env.action_space.low)
# env = gym.make('Pong-v0')
# env = DummyVecEnv([lambda: env])

model = SAC(LnCnnPolicy, env, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()