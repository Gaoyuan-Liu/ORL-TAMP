
import os, sys, time
import math
import pybullet as pb
import numpy as np
from gym import spaces

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')


from utils_h2rl import euclidean_distance



file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion



#######################################################################

class EdgePushPolicyIdeal():
    def __init__(self, scn, pusher):
        # self.initial_state = initial_state
        self.type = 'edgepush'
        self.scn = scn
        self.pusher = pusher
        
      

        # Trained models
        
    # def pre_statege(self):
    #     # put the object in the initial state: hook and focused object
    def specify(self, a, b, p, lg):
        self.goal = np.array([lg.value[0][0], lg.value[0][1], lg.value[0][2]])    
        self.state = np.array([p.value[0][0], p.value[0][1], p.value[0][2]])
        self.robot = a
        self.focused_object = b

    def apply(self, _):
        pb.setRealTimeSimulation(1)
        x_0 = self.state[0] 
        y_0 = self.state[1]

        r_start = 0.1

        # print(f'\nself.goal = {self.goal}\n')
        # print(f'\nself.state = {self.state}\n')

        theta = math.atan2(self.goal[1] - y_0, self.goal[0] - x_0)

        print(f'\ntheta = {theta}\n')
        

        x_1 = x_0 - r_start * math.cos(theta) 
        y_1 = y_0 - r_start * math.sin(theta)

        # dis = euclidean_distance(np.array([x_1, y_1]), np.array([x_0, y_0]))

        dis = euclidean_distance(np.array([self.goal[0], self.goal[1]]), np.array([x_0, y_0]))
        dis = dis - 0.05

        x_2 = x_0 + dis * math.cos(theta)
        y_2 = y_0 + dis * math.sin(theta)


        self.pusher.push(self.robot, [x_1, y_1], [x_2, y_2], [])
        time.sleep(2)

        if pb.getBasePositionAndOrientation(self.focused_object)[0][2] < 0.3:
            raise Exception(" Fall down! ")
        pb.setRealTimeSimulation(0)








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
