from gymnasium.envs.registration import register

register(
    id="gymnasium_env_snake/GridWorld-v0",
    entry_point="gymnasium_env_snake.envs:GridWorldEnv",
)
