import torch
import numpy as np
from env_flexible import RetrieveEnv
import pandas as pd
import time
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DDPG, SAC
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
import pybullet as pb





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():

    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path="./logs/",
        name_prefix="rl_model",
        # save_replay_buffer=True,
        # save_vecnormalize=True,
        )
    
    env = RetrieveEnv() #make_vec_env("Hooking-v0", n_envs=1)#gym.make('Hooking-v0')#HookingEnv() #gym.make('Hooking-v0')

    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=100, batch_size=50, n_epochs=2, verbose=1, tensorboard_log="./tensorboard/")
    model = SAC('MlpPolicy', env, learning_rate=3e-4, batch_size=100, verbose=1, ent_coef='auto_0.1', tensorboard_log="./tensorboard/")   
    model.learn(total_timesteps=20000, callback=checkpoint_callback)
    model.save("./sac_retrieve")

    del model # remove to demonstrate saving and loading

#########################################################################################

def retrain():
    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path="./logs/",
        name_prefix="rl_model",
        # save_replay_buffer=True,
        # save_vecnormalize=True,
        )
    
    env = RetrieveEnv() #make_vec_env("Hooking-v0", n_envs=1)
    # env.connect()
    model = SAC.load("sac_hooking", env=env, tensorboard_log="./tensorboard/")
    model.set_env(env)
    env.reset()
    model.learn(total_timesteps=40000, callback=checkpoint_callback, reset_num_timesteps=False)
    model.save("./sac_hooking_1")

    del model


#########################################################################################

def evaluation():
    # Data initialization
    log_f = open("log_valid.txt","w+")
    success_n = 0
    df = pd.DataFrame(columns=['x_o', 'y_o', 'x_h', 'y_h', 'yaw_h', 'success'])
    goals = []

    # Load model
    env = HookingEnv()
    file = "./preTrained/hook/sac_hooking.zip"
    model = SAC.load(file, env=env, tensorboard_log="./tensorboard/")


    # Episodes
    for eps in range(200):
        init_state = env.reset()
        state = init_state.reshape(-1)


        
            

        while True:

            dis = np.np.linalg.norm(state[:2] - np.array([0,0]))
            if dis < 0.65 and dis > 0.35:
                print("No need to hook")
                data = np.append(init_state, 0)
                break

            action, _states = model.predict(state, deterministic=True)

            print(f'\n action = {action}')
            state, reward, dones, info = env.step(action)
            if dones == True:
                if reward >= 10:
                    print("Success")
                    data = np.append(init_state, 1) # Hindsight goal
                    goals.append(state[:2])
                    success_n += 1
                else:
                    print("Fail")
                    data = np.append(init_state, 0)
                break    


        print(f'data: {data}')
        df.loc[len(df)] = data

        log_f.write('{}\n'.format(success_n))
        log_f.flush()
        df.to_csv('./state_discriminator/data.csv', index=False)
        init_state = env.reset()

                

    env.close()
    


if __name__ == '__main__':
    train()
    # retrain()
    # evaluation()
 
