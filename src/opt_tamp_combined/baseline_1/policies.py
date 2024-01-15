
import os, sys, time
import math
import pybullet as p
import numpy as np
from gym import spaces

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')

from HAC import Agent
from env import EdgePushingEnv
from utils_h2rl import euclidean_distance
from cartesian_planner import CartesianPlanner
from control import Control
from pushing import Pushing


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion




# class EdgePushingProxyEnv():
#     def __init__(self, scn):

#         self.scn = scn
#         self.robot = scn.panda
#         # self.table = scn.table
#         self.dish = scn.dish
#         self.objects = []
#         self.objects.append(self.dish)

#         self.workspace = np.array([[0.25, 0.75], [-0.25, 0.25]])

#         self.pandaNumDofs = 7

#         self.action_space = spaces.Box(low=np.array([-1, -1], dtype=np.float64), high=np.array([1, 1], dtype=np.float64))
#         self.pusher = Pushing()
#         self.observation_space = spaces.Box(low=np.array([0.25, -0.25, 0], dtype=np.float64), high=np.array([0.75, 0.25, 0.5], dtype=np.float64))
    

#         # Goal
#         self.goal = np.array([0.5, 0.3, 0.0])

#         # Debug
#         self.total_steps = 0
        
        

#     def reset(self, goal):
#         p.changeDynamics(self.dish, -1, lateralFriction=1)
#         self.goal = goal
        

#     def step(self, action):
#         self.render(action)
#         observation = self.observe()
#         done = self.done(observation)
#         reward = 0

#         return observation, reward, done, None

#     def render(self, action):

#         x_0 = self.observe()[0] 
#         y_0 = self.observe()[1]

#         r_start = 0.1
#         l_max = 0.4

#         x_1 = r_start * math.cos(action[0] * math.pi) + x_0
#         y_1 = r_start * math.sin(action[0] * math.pi) + y_0

#         dis = euclidean_distance(np.array([x_1, y_1]), np.array([x_0, y_0]))

#         x_2 = x_1 + l_max * (x_0 - x_1) / dis * (action[1] + 1) / 2
#         y_2 = y_1 + l_max * (y_0 - y_1) / dis * (action[1] + 1) / 2



#         self.pusher.push(self.robot, [x_1, y_1], [x_2, y_2], self.objects)

#     def done(self, obs):
#         done = False
#         if abs(obs[1] - self.goal[1]) <= 0.05:
#             done = True
#         return done
    
#     def observe(self):
#         pos, orn = p.getBasePositionAndOrientation(self.dish)
#         # rpy = p.getEulerFromQuaternion(orn)
#         return np.array([pos[0], pos[1], pos[2]], dtype=np.float64)

    
#     def render_goal(self, goal):
#         goal = goal.reshape((-1,))
#         position = [goal[0], goal[1], 0.0005] 
#         orn = [0,0,0,1]#quaternion_from_euler(0, 0, goal[2])
#         # p.resetBasePositionAndOrientation(self.scn.goal_surface, position, orn)


    
#######################################################################

class EdgePushPolicyIdeal():
    def __init__(self, scn, pusher):
        # self.initial_state = initial_state
        self.type = 'hook'
        self.scn = scn
        self.pusher = pusher
        
      

        # Trained models
        


    # def pre_statege(self):
    #     # put the object in the initial state: hook and focused object
    def specify(self, a, p, lg):
        self.goal = np.array([lg.value[0][0], lg.value[0][1], lg.value[0][2]])    
        self.state = np.array([p.value[0][0], p.value[0][1], p.value[0][2]])
        self.robot = a

    def apply(self, _):
        p.setRealTimeSimulation(1)
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
