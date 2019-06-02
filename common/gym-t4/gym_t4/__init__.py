from gym.envs.registration import register

register(
    id='t4-v0',
    max_episode_steps=20,
    entry_point='gym_t4.envs:T4Env',
)