import torch
import gym
import numpy as np
import math
from env import HookingEnv
import wandb
from utils_h2rl import WindowMean

import pandas as pd
import time
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DDPG
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():

    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path="./logs/",
        name_prefix="rl_model",
        # save_replay_buffer=True,
        # save_vecnormalize=True,
        )
    
    env = make_vec_env("Hooking-v1", n_envs=1)#gym.make('Hooking-v0')#HookingEnv() #gym.make('Hooking-v0')

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=100, batch_size=50, n_epochs=2, verbose=1, tensorboard_log="./tensorboard/")
    model = DDPG('MlpPolicy', env, learning_rate=3e-4, batch_size=50, verbose=1, tensorboard_log="./tensorboard/", action_noise=action_noise)   
    model.learn(total_timesteps=20000, callback=checkpoint_callback)
    model.save("./ddpg_hooking")

    del model # remove to demonstrate saving and loading


def retrain():
    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path="./logs/",
        name_prefix="rl_model",
        # save_replay_buffer=True,
        # save_vecnormalize=True,
        )
    
    
    env = make_vec_env("Hooking-v1", n_envs=1)
    # env.connect()
    model = DDPG.load("ddpg_hooking", env=env, tensorboard_log="./tensorboard/")
    model.set_env(env)
    env.reset()
    model.learn(total_timesteps=20000, callback=checkpoint_callback, reset_num_timesteps=False)
    model.save("./ddpg_hooking_1")

    del model

def evaluation():
    # Data initialization
    log_f = open("log_valid.txt","w+")
    success_n = 0
    df = pd.DataFrame(columns=['x_o', 'y_o', 'x_h', 'y_h', 'yaw_h', 'success'])
    goals = []

    # Load model
    env = make_vec_env("Hooking-v1", n_envs=1)
    file = "./preTrained/hook/ddpg_hooking.zip"
    model = DDPG.load(file, env=env, tensorboard_log="./tensorboard/")

    # Evaluation
    init_state = env.reset()
    state = init_state
    for i in range(1000):
        action, _states = model.predict(state, deterministic=True)
        state, rewards, dones, info = env.step(action)
        # env.render()
        if dones == True:
            if rewards >= 10:
                print("Success")
                data = np.append(init_state, 1) # Hindsight goal
                goals.append(state[:2])
                success_n += 1
            else:
                print("Fail")
                data = np.append(init_state, 0)

            print(f'data: {data}')
            df.loc[len(df)] = data

            log_f.write('{}\n'.format(success_n))
            log_f.flush()
            df.to_csv('./state_discriminator/data_cube.csv', index=False)
            init_state = env.reset()

    env.close()
    


if __name__ == '__main__':
    # train()
    # retrain()
    evaluation()
 
