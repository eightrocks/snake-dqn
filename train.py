import gymnasium
import gymnasium_env_snake
from stable_baselines3 import DQN

env = gymnasium.make('gymnasium_env_snake/GridWorld-v0')

model = DQN(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    batch_size=64,
    buffer_size=50000,
    learning_starts=1000,
    train_freq=1,
    gradient_steps=1,
    exploration_fraction=0.3,
    verbose=1
)
model.learn(total_timesteps=200000, log_interval=4)
model.save("dqn_snake")

del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_snake")

# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()

