
import os, sys, time
import math
import pybullet as p
import numpy as np
from gym import spaces

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')

# from agent import Agent
from env import HookingEnv
# from utils_h2rl import euclidean_distance
# from cartesian_planner import CartesianPlanner
# from control import Control


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, DDPG




class HookEnvNominal(HookingEnv):
    def __init__(self, scn):
        super(HookEnvNominal, self).__init__()
        self.scn = scn
        self.robot = scn.panda
        self.hook = scn.hook

        
    def reset(self, focused_object):
        self.focused_object = focused_object
        self.objects = [self.focused_object, self.hook]
        state = self.observe()
        self.episode_length = 0
        self.pre_dis = 0 #euclidean_distance(observation[:2], self.goal[:2])
        return state
    
    def connect(self):
        pass
    
  
    
#######################################################################

class HookingPolicy():
    def __init__(self, scn):
        # self.initial_state = initial_state
        self.type = 'hook'
        self.scn = scn
        self.env = HookEnvNominal(scn)

        # Load model
        # env = make_vec_env("Hooking-v0", n_envs=1)
        file = "./preTrained/ddpg_hooking"
        self.model = DDPG.load(file, env=self.env)
       

    def specify(self, b, lg):
        self.focused_object = b
        # self.goal = np.array([lg.value[0][0], lg.value[0][1], 0])    

    def apply(self, _):
        p.setRealTimeSimulation(1)
        state = self.env.reset(self.focused_object)
        while True:
            action, _states = self.model.predict(state, deterministic=True)
            state, rewards, dones, info = self.env.step(action)

            if dones == True:
                break

        p.setRealTimeSimulation(0)





#######################################################################

class Observe():
    def __init__(self):
        self.type = 'observe'
        # self.scn = scn

    def apply(self, _):
        p.setRealTimeSimulation(1)
        p.setRealTimeSimulation(0)


#######################################################################

# def main():
