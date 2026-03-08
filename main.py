import gymnasium
import gymnasium_env_snake
from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo
import os

env = gymnasium.make('gymnasium_env_snake/GridWorld-v0', render_mode='rgb_array')

model = DQN.load("dqn_snake")







# while True:
#     action, _states = model.predict(obs, deterministic=False)
#     obs, reward, terminated, truncated, info = env.step(action)
#     while not (terminated or truncated):
#         action, _states = model.predict(obs, deterministic=False)
#         obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#          obs, info = env.reset()


env = RecordVideo(env, video_folder="./save_videos1", video_length=10000,episode_trigger=lambda x: x==0,disable_logger=True)

obs, info = env.reset()

for i in range(100):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)
    # while not (terminated or truncated):
    #     action, _states = model.predict(obs, deterministic=False)
    #     obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
         obs, info = env.reset()

env.close()
