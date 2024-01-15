from __future__ import print_function

import copy
import pybullet as pb
import random
import time, os, sys
from itertools import islice, count
import math

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')

from examples.pybullet.utils.pybullet_tools.panda_primitives import Pose

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
from utils_h2rl import euclidean_distance





##################################################

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../state_discriminator/')

##################################################

def get_loose_goal_push_gen(problem, collisions=True, **kwargs):
    def gen(body):
        while True:
            # goal = random.choice(goals)
            # print(f'\ngoal = {goal}')
            goal_x = np.random.uniform(low=0.25, high=0.75)
            goal_y = random.choice([-0.24, 0.24])
            body_pose = ((goal_x, goal_y, 0.31), (0.0, 0.0, 0.0, 1.0))
            # body_pose = ((0.5, 0.24, 0.31), (0.0, 0.0, 0.0, 1.0))
            ap = Pose(body, body_pose)
            yield (ap,)
    return gen


##################################################
def get_pushing_gen(problem, pusher, collisions=True, **kwargs):
    # pushing = Push(a, obj, p, lg, action_list, pusher)
    def gen(a, obj, p, lg):
        action_list = []
        pushing = Push(a, obj, p, lg, pusher)
        pushing.action_list = []
        # while True:
        gripper = problem.get_gripper()
        pb.resetBasePositionAndOrientation(gripper, (10, 10, 0.2), (0,0,0,1))
        pb.setRealTimeSimulation(1)

        start = time.time()
        action = np.zeros(2)
        success = False
        pb.resetBasePositionAndOrientation(obj, (0.5, 0, 0.31), (0,0,0,1))
        for i in range(20):
            state = pb.getBasePositionAndOrientation(obj)[0]
            # Fall reset
            if state[2] < 0.3:
                pb.resetBasePositionAndOrientation(obj, (0.5, 0, 0.31), (0,0,0,1))
                pushing.action_list = []
            
            action[0] = np.random.uniform(low=-1, high=1)
            action[1] = np.random.uniform(low=-1, high=1)
            
            pushing.render(state, action, [obj])
            state = pb.getBasePositionAndOrientation(obj)[0]
            pushing.action_list.append(action)

            # if euclidean_distance(np.array([state[0], state[1]]), np.array([lg.value[0][0], lg.value[0][1]])) <= 0.05:
            if abs(state[1] - lg.value[0][1]) <= 0.05:
                success = True
                break
                
            # Time out
            end = time.time()
            if end - start > 10:
                success = False
                break
                

        pb.setRealTimeSimulation(0)   
        if success == True:       
            yield (pushing, )
           

    return gen

##################################################
class Push():
    def __init__(self, robot, obj, p, g, pusher):
        self.robot = robot
        self.obj = obj
        self.p = p
        self.g = g
        self.action_list = []
        self.pusher = pusher
        self.type = 'push'
    def apply(self, _):
        pb.setRealTimeSimulation(1)
        state = self.p.value[0]
        for action in self.action_list:
            self.render(state, action, [self.obj])
            state = pb.getBasePositionAndOrientation(self.obj)[0]
        pb.setRealTimeSimulation(0)
    def render(self, state, action, objects):
        x_0 = state[0] 
        y_0 = state[1]

        r_start = 0.1
        l_max = 0.4

        x_1 = r_start * math.cos(action[0] * math.pi) + x_0
        y_1 = r_start * math.sin(action[0] * math.pi) + y_0

        dis = euclidean_distance(np.array([x_1, y_1]), np.array([x_0, y_0]))

        x_2 = x_1 + l_max * (x_0 - x_1) / dis * (action[1] + 1) / 2
        y_2 = y_1 + l_max * (y_0 - y_1) / dis * (action[1] + 1) / 2

        # x_2 =  x_1 - l_max * math.cos(action[0] * math.pi) * action[1]
        # y_2 =  y_1 - l_max * math.sin(action[0] * math.pi) * action[1]

        # print(f'\n {x_1}, {y_1}, {x_2}, {y_2}')

        self.pusher.push(self.robot, [x_1, y_1], [x_2, y_2], objects)
        

    



##################################################

# def render(pusher, robot, state, action, objects):
    
#     x_0 = state[0] 
#     y_0 = state[1]

#     r_start = 0.1
#     l_max = 0.4

#     x_1 = r_start * math.cos(action[0] * math.pi) + x_0
#     y_1 = r_start * math.sin(action[0] * math.pi) + y_0

#     dis = euclidean_distance(np.array([x_1, y_1]), np.array([x_0, y_0]))

#     x_2 = x_1 + l_max * (x_0 - x_1) / dis * (action[1] + 1) / 2
#     y_2 = y_1 + l_max * (y_0 - y_1) / dis * (action[1] + 1) / 2

#     # x_2 =  x_1 - l_max * math.cos(action[0] * math.pi) * action[1]
#     # y_2 =  y_1 - l_max * math.sin(action[0] * math.pi) * action[1]

#     # print(f'\n {x_1}, {y_1}, {x_2}, {y_2}')

#     pusher.push(robot, [x_1, y_1], [x_2, y_2], objects)



##################################################

def get_same_surface_test(problem, collisions=True):
    def test(o, p, gl): # Maybe now only define as if they have the same height
        # if abs(p.value[0][2] - gl.value[0][2]) >= 0.01:
        #     return False
        if p.support != gl.support:
            return False
        return True
    return test

##################################################
def get_push_initial_test(problem, collisions=True):


    def test(b, p):
        
        # p.assign()
        # input = np.array([p.value[0][0], p.value[0][1], 0])
        # input = torch.tensor(input,dtype=torch.float32)
        # model.eval()
        # output = model(input).detach().numpy().round()

        # if output == 1:
        #     return True
        # else:
        #     return False
        if p.value[0][0] >= 0.7:
            return False
        else:
            return True

    return test



def main():
    gen = get_loose_goal_gen(None, collisions=True)
    for i in range(5):
        # gen(None)
        print(next(gen(None))[0].value)








if __name__ == '__main__':
    main()

