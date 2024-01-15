from stable_baselines3.common.env_checker import check_env
from env_vision import EdgePushingEnv


env = EdgePushingEnv()
check_env(env, warn=True)
