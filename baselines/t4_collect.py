from t4_env import T4Env


env = T4Env(metrics_key='c_1')

obs = env.reset()

for x in range(50000):
    action = env.action_space.sample()
    obs, _, _, _ = env.step(action)