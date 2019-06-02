from stable_baselines import ACER
from stable_baselines.gail import ExpertDataset
from t4_env import T4Env
from stable_baselines.common.vec_env import DummyVecEnv
# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset
dataset = ExpertDataset(expert_path='t4_dataset.npz',
                        traj_limitation=1, batch_size=1)

env = DummyVecEnv([lambda: T4Env()])

model = ACER('CnnPolicy', env, verbose=1) #, nminibatches=1)
# Pretrain the PPO2 model
model.pretrain(dataset, n_epochs=1000)

model.save('acer-t4-pretrain')