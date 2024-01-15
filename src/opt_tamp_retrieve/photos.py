
#! /venv3.8/bin/python
from email.policy import default
import os, sys
from tokenize import Double
import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np
import math
import pybullet as pb
import time
import colorama
from colorama import Fore
import pandas as pd
import cv2

# Import Searching Dict
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../pddlstream/')
sys.path.insert(0, file_path)

from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion
from examples.pybullet.utils.pybullet_tools.utils import connect, wait_for_user

from control import Control
sys.path.insert(0, file_path + '/../')
from common.scn import Scenario


# Global
pandaNumDofs = 7
initialJointPoses = (0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0)


class HookingEnv(gym.Env):
    def __init__(self):
        super(HookingEnv, self).__init__()

        # Mode
        self.mode = 'train'
        
        # Action space
        # self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float64), high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float64))
        self.action_space = spaces.Box(low=np.array([-1, -1, -1], dtype=np.float64), high=np.array([1, 1, 1], dtype=np.float64))

        # State space
        # [cup, hook]
        self.observation_space = spaces.Box(low=np.array([0, -1.2, -0.5, -1.2, -3.14]), 
                                            high=np.array([1.2, 1.2, 1.2, 1.2, 3.14]),
                                            dtype=np.float64)


        self.control = Control()

        # Goal
        self.goal = np.array([0.5, 0.3, 0.0])

        # Debug
        self.total_steps = 0
        self.connect()



    ####################################################
    def connect(self, use_gui=True):
        self.sim_id = connect(use_gui=True)
        self.scn = Scenario()
        # pb.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
        # pb.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # pb.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        # pb.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        # pb.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

        # Objects
        self.scn.reset()
        self.robot = self.scn.add_robot(pose=((0,0,0.0), (0,0,0,1)))
        self.control.finger_open(self.robot)

        # self.table = self.scn.add_low_table()

        self.objects = []
        cup_1 = self.scn.add_cup()
        cup_2 = self.scn.add_cup()

        self.scn.add_hook()
        self.scn.add_surface()
 
        self.objects += self.scn.cups
        self.objects += self.scn.hooks
        self.target_1 = cup_1
        self.target_2 = cup_2



        # self.goal_mark = self.scn.add_goal_mark()
        
    ####################################################

    def reset(self):
        if self.mode == 'train':
            obs = self.reset_train()
            
        elif self.mode == 'eval':
            obs =  self.reset_eval()

        return obs
         
    
    ####################################################
    def reset_train(self):
        print("\033[92m {}\033[00m" .format(f'\n New Episode'))

        # Reset Robot
        for i in range(pandaNumDofs):
            pb.setJointMotorControl2(self.robot, i, pb.POSITION_CONTROL, initialJointPoses[i],force=5 * 240.)
        
        # ------------------------------
        # Reset Objects
        r = np.random.uniform(low=0.8, high=1.1)
        theta = np.random.uniform(low=-math.pi/6, high=math.pi/6)
        x = np.random.uniform(low=0.8, high=1.1) #r*math.cos(theta)
        y = np.random.uniform(low=-0.2, high=0.2) #r*math.sin(theta)
        f_coeff = 1 #np.random.uniform(low=0.4, high=1.0)
        
        z = 0.025 
        yaw = np.random.uniform(low=-math.pi, high=math.pi)
        orn = quaternion_from_euler(0,0,yaw)
        pb.resetBasePositionAndOrientation(self.objects[0], [x, y, z], orn)
        # Change friction
        
        pb.changeDynamics(self.objects[0], -1, lateralFriction=f_coeff)

        # ------------------------------
        # Reset Hook
        # If hook is not reset, then the training will stuck in dead-end  
        r_hook = np.random.uniform(low=0.4, high=0.7)
        theta_hook = np.random.uniform(low=-math.pi/6, high=math.pi/6)
        x_hook = np.random.uniform(low=0.35, high=0.6) #r_hook*math.cos(theta_hook)
        y_hook = np.random.uniform(low=-0.2, high=0.2) #r_hook*math.sin(theta_hook)
       
        yaw_hook = np.random.uniform(low=-math.pi/4, high=math.pi/4)
        orn_hook = quaternion_from_euler(0,0,yaw_hook)
        pb.resetBasePositionAndOrientation(self.objects[-1], (x_hook, y_hook, 0.02), orn_hook)

        # ------------------------------
        # Reset Goal
        r_goal = np.random.uniform(low=0.4, high=0.6)
        theta_goal = np.random.uniform(low=-math.pi/4, high=math.pi/4)
        x_goal = r_goal*math.cos(theta_goal)
        y_goal = r_goal*math.sin(theta_goal)
        # pb.resetBasePositionAndOrientation(self.goal_mark, (x_goal, y_goal, 0.151), (0,0,0,1))
        self.goal = np.array([x_goal, y_goal, 0.151])
        # self.goal = np.array([np.random.uniform(low=0.3, high=0.6), np.random.uniform(low=-0.4, high=0.4), 0.0])
        
        # ------------------------------
        pb.setRealTimeSimulation(1)
        self.observation = self.observe()

        self.pre_dis = np.linalg.norm(self.observation[:2] - self.goal[:2])

    
        return self.observation
    
    ####################################################
    def reset_eval(self):
        pb.setRealTimeSimulation(1)
        print("\033[92m {}\033[00m" .format(f'\n New Episode'))

        # Reset Robot
        for i in range(pandaNumDofs):
            pb.setJointMotorControl2(self.robot, i, pb.POSITION_CONTROL, initialJointPoses[i],force=5 * 240.)
        
        # ------------------------------
        # Reset Objects
        x = np.random.uniform(low=0.0, high=1)
        y = np.random.uniform(low=-1, high=1)
        f_coeff = np.random.uniform(low=0.4, high=1.0)
        
        z = 0.025
        yaw = np.random.uniform(low=-math.pi, high=math.pi)
        orn = quaternion_from_euler(0,0,yaw)
        pb.resetBasePositionAndOrientation(self.objects[0], [x, y, z], orn)
        # Change friction
        
        pb.changeDynamics(self.objects[0], -1, lateralFriction=f_coeff)

        # ------------------------------
        # Reset Hook
        x_hook = 0.5
        y_hook = 0.0
        yaw_hook = np.random.uniform(low=-math.pi/4, high=math.pi/4)
        orn_hook = quaternion_from_euler(0,0,yaw_hook)
        orn_hook = quaternion_from_euler(0,0,0)
        pb.resetBasePositionAndOrientation(self.objects[-1], (x_hook, y_hook, 0), orn_hook)

        # ------------------------------
        # Reset Goal
        self.goal = np.array([np.random.uniform(low=0.3, high=0.6), np.random.uniform(low=-0.4, high=0.4), 0.0])
        self.observation = self.observe()
        self.pre_dis = np.linalg.norm(self.observation[:2] - self.goal[:2])

        self.episodic_return = 0
        self.episode_length = 0
        self.total_steps = 0

        return self.observation


    ####################################################

    def step(self, action):

        print("\033[30m {}\033[00m" .format(f'\n {self.episode_length + 1} Step'))

        self.render(action)
        self.observation = self.observe()

        reward, done = self.reward(self.observation)

        # Data
        self.episodic_return += reward
        self.episode_length += 1
        self.total_steps += 1
        
        # Done
        if self.episode_length >= 20:
            done = True

        return self.observation, reward, done, {}


    ####################################################

    def observe(self):
        total = []
        # for object in self.objects:
        pos, orn = pb.getBasePositionAndOrientation(self.target)
        rpy = pb.getEulerFromQuaternion(orn)
        total.append(pos[0])
        total.append(pos[1])
        # total.append(rpy[-1])

        pos, orn = pb.getBasePositionAndOrientation(self.scn.hook)
        rpy = pb.getEulerFromQuaternion(orn)
        total.append(pos[0])
        total.append(pos[1])
        total.append(rpy[-1])

        obs = np.array(total)
        
        return obs

    ####################################################

    def render(self, action):
        pass




    ####################################################


    def reward(self, obs):
        done = False
        reward = 0
        
        dis = np.linalg.norm(obs[:2] - np.array([0,0])) #self.goal[:2])
        
        hook_pose = pb.getBasePositionAndOrientation(self.objects[-1])
        object_pose = pb.getBasePositionAndOrientation(self.objects[0])


        if hook_pose[0][-1] < 0.15:
            return -5, True
        
        if object_pose[0][-1] < 0.15:
            return -5, True


        get_closer = self.pre_dis - dis
        if abs(get_closer) > 0.01: # Block is touched
            reward += 1

        if dis < 0.6 and dis > 0.35:
        # if dis < 0.1:
            done = True
            reward += 20
        else:
            reward -= 1    
        self.pre_dis = dis
        print(f' reward = {reward}')
        return reward, done


    def render_goal(self, goal):
        goal = goal.reshape((-1,))
        position = [goal[0], goal[1], 0.0005] 
        orn = [0,0,0,1]#quaternion_from_euler(0, 0, goal[2])
        # pb.resetBasePositionAndOrientation(self.scn.cube_shadow, pos, orn)
        pb.resetBasePositionAndOrientation(self.scn.goal_mark, position, orn)

    
    # This step is for sub-goals which are solved by solver
    def solved_step(self):
        observation = self.observe()
        _, done = self.reward(observation)
        reward = 10
        info = None
        # print(f'The reward from env.solved_step() is {reward}')
        return observation, reward, done, info
    
    # def flip_checking(self, obj_poses):
    #     for i in obj_poses:
    #         eulers = get_euler(i)
    #         new_eulers = list(eulers)
    #         if abs(eulers[0]) >= 0.5:
    #             new_eulers = [0.0, eulers[1], eulers[2]]
                
    #         if abs(eulers[1]) >= 0.5:
    #             new_eulers = [eulers[0], 0.0, eulers[2]]
    #         set_euler(i, new_eulers)

########################################################
# Tools
########################################################
    
def take_a_photo():
    viewMatrix = pb.computeViewMatrix(
                # cameraEyePosition=[pos[0], pos[1], 0.4],
                # cameraTargetPosition=[pos[0], pos[1], 0],
                cameraEyePosition=[1, 1, 1],
                cameraTargetPosition=[0.5, 0, 0.5],
                cameraUpVector=[0, 0, 1])
    near = 0.01
    far = 5
    projectionMatrix = pb.computeProjectionMatrixFOV(
                    fov= 86,
                    aspect= 1,
                    nearVal= near,
                    farVal=far)
    
    img = pb.getCameraImage(width=1000, 
                            height=1000,
                            viewMatrix=viewMatrix,
                            projectionMatrix=projectionMatrix, 
                            renderer=pb.ER_BULLET_HARDWARE_OPENGL)
    image_bgr = cv2.cvtColor(img[2], cv2.COLOR_RGB2BGR)
    return image_bgr
            


        
if __name__ == '__main__':
    # sim_id = connect(use_gui=True)

    env = HookingEnv()

    pb.setRealTimeSimulation(0)
    wait_for_user()

    yaw_hook = np.random.uniform(low=-math.pi/4, high=math.pi/4)
    orn_hook = quaternion_from_euler(0,0,yaw_hook)
    pb.resetBasePositionAndOrientation(env.objects[-1], (0.4, .05, 0.02), orn_hook)
    # pb.resetBasePositionAndOrientation(env.objects[-1], (0.4, -0.1, 0.02), (0,0,0,1))
    pb.resetBasePositionAndOrientation(env.target_1, (0.9, -0.1, 0.04), (0,0,0,1))
    pb.resetBasePositionAndOrientation(env.target_2, (0.6, 0.45, 0.04), (0,0,0,1))
    wait_for_user()

    img = take_a_photo()
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imwrite('./output_image.jpg', img)
    
    
    # action = np.array((-0.7,1,1))
    # env.render(action)

    # wait_for_user()

    # action = np.array((0.7,1,1))
    # env.render(action)

    # observation_space = env.observation_space

    # print(observation_space.sample())

    wait_for_user()





