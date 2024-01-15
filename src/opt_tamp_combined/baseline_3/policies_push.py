
import os, sys, time
import math
import pybullet as p
import numpy as np
from gym import spaces

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../')
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../opt_tamp_push/')
from opt_tamp_push.env import EdgePushingEnv
from opt_tamp_push.pushing import Pushing
from opt_tamp_push.utils_h2rl import euclidean_distance


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, DDPG



class EdgePushingEnvNominal(EdgePushingEnv):
    def __init__(self, scn):
        super(EdgePushingEnvNominal, self).__init__()

        self.scn = scn
        self.robot = scn.panda
        # self.table = scn.table
        self.dish = scn.dish
        
   

    def reset(self, focused_object, goal):
        p.changeDynamics(self.dish, -1, lateralFriction=1)
        self.goal = goal
        self.focused_object = focused_object
        self.objects = [self.focused_object]
        state = self.observe()
        self.pre_dis = euclidean_distance(state[0:2], self.goal[0:2])
        self.episodic_return = 0
        self.episode_length = 0
        return state

    def connect(self):
        self.pusher = Pushing()

    

  
    
#######################################################################

class EdgePushPolicy():
    def __init__(self, scn):
        # self.initial_state = initial_state
        self.type = 'hook'
        self.scn = scn
        self.env = EdgePushingEnvNominal(scn)

        file = "./preTrained/edgepush/ddpg_edgepushing"
        self.model = DDPG.load(file, env=self.env)
       
    
    def specify(self, b, lg):
        self.focused_object = b
        # self.goal = np.array([lg.value[0][0], lg.value[0][1], 0])   
        self.goal = np.array([lg[0], lg[1], 0.31])
        


    def apply(self, _):
        p.setRealTimeSimulation(1)
        state = self.env.reset(self.focused_object, self.goal)
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
