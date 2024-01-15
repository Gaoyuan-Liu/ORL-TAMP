import torch
import gym
import numpy as np
import math
from env_ee import EdgePushingEnv

import pandas as pd
import time
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DDPG, SAC
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
    
    env = EdgePushingEnv() #make_env("EdgePushing-v1", n_envs=1) 

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=400, batch_size=200, n_epochs=1, verbose=1, tensorboard_log="./tensorboard/")
    model = SAC('MlpPolicy', env, learning_rate=3e-4, batch_size=200, learning_starts=500, ent_coef='auto_0.1', tensorboard_log="./tensorboard/")   
    # model = PPO('MlpPolicy', env, learning_rate=3e-5, n_steps=100, batch_size=50, n_epochs=2, verbose=1, tensorboard_log="./tensorboard/")#, action_noise=action_noise) #, policy_kwargs=dict(normalize_images=False))
    model.learn(total_timesteps=40000, callback=checkpoint_callback)
    model.save("./sac_edgepushing")

    del model # remove to demonstrate saving and loading


def retrain():
    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path="./logs/",
        name_prefix="rl_model",
        # save_replay_buffer=True,
        # save_vecnormalize=True,
        )
    
    
    # env = make_vec_env("EdgePushing-v1", n_envs=1)
    env = EdgePushingEnv()
    # env.connect()
    model = SAC.load("ppo_edgepush", env=env, tensorboard_log="./tensorboard/")
    model.set_env(env)
    env.reset()
    model.learn(total_timesteps=40000, callback=checkpoint_callback, reset_num_timesteps=False)
    model.save("./ppo_edgepush_1")

    del model

def evaluation():
    # Data initialization
    log_f = open("log_valid.txt","w+")
    success_n = 0
    # df = pd.DataFrame(columns=['x_o', 'y_o', 'x_g', 'y_g', 'success'])
    df = pd.DataFrame(columns=['image', 'success'])
    goals = []

    # Load model
    env = EdgePushingEnv() # make_vec_env("EdgePushing-v0", n_envs=1)
    file = "./trained/with_2_as_punishment/rl_model_30500_steps"
    model = SAC.load(file, env=env, tensorboard_log="./tensorboard/")

    # Evaluation
    init_state = env.reset()
    state = init_state

    X = []
    y = []
    for i in range(2000):
        action, _states = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        if done == True:
            if reward >= 5:
                # data = np.array([init_state, 1]) # Hindsight goal
                X.append(init_state)
                y.append(1)
                success_n += 1
            else:
                print(" Fail")
                # data = np.array([init_state, 0])
                X.append(init_state)
                y.append(0)

            log_f.write('{}\n'.format(success_n))
            log_f.flush()
            # df.to_csv('./state_discriminator/data/data_image.csv', index=False)
            init_state = env.reset()
            # input("Press Enter to continue...")

            np.savez('./state_discriminator/data/data_image.npz', features=X, labels=y)


    env.close()
    


if __name__ == '__main__':
    train()
    # retrain()
    # evaluation()
 
