from __future__ import print_function

import copy
import pybullet as pb
import random
import time, os, sys
from itertools import islice, count

import numpy as np
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')

from examples.pybullet.utils.pybullet_tools.panda_primitives import Pose
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
from utils_h2rl import euclidean_distance
from motion_planner import CartesianPlanner
from control import Control


##################################################

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../state_discriminator/')
from utils import save_list, load_list



##################################################

def get_loose_goal_gen(problem, collisions=True, **kwargs):
    goals = load_list('./../state_discriminator/goals')
    def gen(body):
        while True:
            goal = random.choice(goals)
            body_pose = ((goal[0], goal[1], 0.026), (0.0, 0.0, 0.0, 1.0))
            ap = Pose(body, body_pose)
            yield (ap,)
    return gen

##################################################


def get_hook_initial_test(problem, collisions=True):
    

    def test(b, p):
        dis = euclidean_distance(p.value[0][:2], np.array([0, 0]))
        if dis > 0.8:
            return True
        # p.assign()
        # input = np.array([p.value[0][0], p.value[0][1], 0])
        # input = torch.tensor(input,dtype=torch.float32)
        # model.eval()
        # output = model(input).detach().numpy().round()

        # if output == 1:
        #     return True
        # else:
        #     return False
        return False
    return test


##################################################
def get_hooking_gen(problem, collisions=True, **kwargs):

    planner = CartesianPlanner()
    controller = Control()
    tool = problem.tools[0]   
    
    def gen(a, obj, p, lg):
        
        # while True:
        tool_pose = pb.getBasePositionAndOrientation(tool)

        hooking = Hook(a, obj, p, lg, tool, planner, controller)

        for i in range(5):
            
            pb.setRealTimeSimulation(1)
            action = np.zeros(3)

            # Reset
            p.assign()
            pb.resetBasePositionAndOrientation(tool, tool_pose[0], tool_pose[1])
            state = p.value[0]
            success = False
            for i in range(5):
                start = time.time()
                # Random action
                action[0] = np.random.uniform(low=-1, high=1)
                action[1] = np.random.uniform(low=-1, high=1)
                action[2] = np.random.uniform(low=-1, high=1)
                
                hooking.render(state, action, [obj])
                state = pb.getBasePositionAndOrientation(obj)[0]
                hooking.action_list.append(action)

                dis = euclidean_distance(state[:2], np.array([0, 0]))
                if dis > 0.35 and dis < 0.6:
                    success = True
                    break
                end = time.time()
                if end - start > 10:
                    success = False
                    break
            pb.setRealTimeSimulation(0)   
            if success == True:       
                yield (hooking, )
           

    return gen

##################################################

class Hook():
    def __init__(self, robot, obj, p, g, tool, planner, controller):
        self.robot = robot
        self.obj = obj
        self.p = p
        self.g = g
        self.action_list = []
        self.cartesian_planner = planner
        self.control = controller
        self.pandaNumDofs = 7
        self.tool = tool
        self.type = 'hook'


    def apply(self, _):
        pb.setRealTimeSimulation(1)
        state = self.p.value[0]
        for action in self.action_list:
            self.render(state, action, [self.obj])
            state = pb.getBasePositionAndOrientation(self.obj)[0]
        pb.setRealTimeSimulation(0)

    def render(self, state, action, objects):
        # Goal position
        # d_theta = action[0] * math.pi/2
        # d_r = 0.55 + action[1] * 0.15
        d_yaw = action[2] * math.pi/2

        d_x = 0.5 + 0.2 * action[0] #d_r * math.cos(d_theta)
        d_y = 0.4 * action[1] #d_r * math.sin(d_theta)

        # ------------------------------

        # Go approach
        info = pb.getBasePositionAndOrientation(self.tool)
        ee_pos1 = list(info[0])
        ee_pos1[-1] = ee_pos1[-1] + 0.14
        hook_orn = info[1]
        hook_rpy = euler_from_quaternion(hook_orn)
        ee_yaw1 = (hook_rpy[-1] - math.pi/4) % (np.sign(hook_rpy[-1] - math.pi/4)*(math.pi))
        ee_orn1 = quaternion_from_euler(math.pi, 0, ee_yaw1)
        approach_pose = (ee_pos1, ee_orn1)

        # Change approach
        if d_yaw > 0 and ee_yaw1 > 0:
            ee_yaw1 -= math.pi
        if d_yaw < 0 and ee_yaw1 < 0:
            ee_yaw1 += math.pi 
        ee_orn1 = quaternion_from_euler(math.pi, 0, ee_yaw1)
        approach_pose = (ee_pos1, ee_orn1)

        # Go grasp (down)
        info = pb.getBasePositionAndOrientation(self.tool)
        ee_pos2 = list(info[0])
        ee_pos2[-1] = ee_pos1[-1] - 0.04
        ee_orn2 = ee_orn1
        grasp_pose = (ee_pos2, ee_orn2)

        # Move
        # observation = self.observe()
        x = d_x
        y = d_y
        yaw_3 = d_yaw #ee_yaw1 + d_yaw 

        ee_orn3 = quaternion_from_euler(math.pi, 0, yaw_3)
        ee_pos3 = np.array([x, y, 0.15]).astype(float)
        goal_pose = (ee_pos3, ee_orn3)

        # Plan & execution
        end_pose_list = [approach_pose, grasp_pose, goal_pose]
        joint_states = pb.getJointStates(self.robot, list(range(self.pandaNumDofs)))# Given by Holding
        start_conf = []
        for i in joint_states:
            start_conf.append(i[0])
        success, trajectory = self.cartesian_planner.plan_multipose(start_conf, end_pose_list)
        for waypoint in trajectory:
            time.sleep(0.01) # Moving speed
            finger_state = pb.getLinkState(self.robot, 11)[4]
            if finger_state[-1] <= 0.05:
                self.control.finger_close(self.robot)
            self.control.go_joint_space(self.robot, waypoint)
        self.control.finger_open(self.robot)
        

        
        # Up
        if success:
            ee_pos4 = ee_pos3
            ee_pos4[-1] = ee_pos4[-1] + 0.1
            ee_orn4 = ee_orn3
            up_pose = (ee_pos4, ee_orn4)
            # info = pb.getBasePositionAndOrientation(self.tool)
            end_pose_list = [up_pose]
            joint_states = pb.getJointStates(self.robot, list(range(self.pandaNumDofs)))# Given by Holding
            start_conf = []
            for i in joint_states:
                start_conf.append(i[0])
            _, trajectory = self.cartesian_planner.plan_multipose(start_conf, end_pose_list)
            for waypoint in trajectory:
                time.sleep(0.01) # Moving speed
                finger_state = pb.getLinkState(self.robot, 11)[4]
                self.control.go_joint_space(self.robot, waypoint)
                self.control.finger_open(self.robot)


        # Reset
        joint_states = pb.getJointStates(self.robot, list(range(self.pandaNumDofs)))# Given by Holding
        start_conf = []
        for i in joint_states:
            start_conf.append(i[0])
        default_config = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
        trajectory = self.cartesian_planner.plan_joint(start_conf, default_config)
        for waypoint in trajectory:
            time.sleep(0.02) # Moving speed
            self.control.go_joint_space(self.robot, waypoint)

##################################################



def main():
    gen = get_loose_goal_gen(None, collisions=True)
    for i in range(5):
        # gen(None)
        print(next(gen(None))[0].value)








if __name__ == '__main__':
    main()

