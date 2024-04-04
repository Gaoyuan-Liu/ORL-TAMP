
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
import matplotlib.pyplot as plt
import copy
from scipy.ndimage import convolve


# Import Searching Dict
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../pddlstream/')
sys.path.insert(0, file_path + '/../utils/')
sys.path.insert(0, file_path + '/../')


from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion
from examples.pybullet.utils.pybullet_tools.utils import connect, wait_for_user, multiply, invert, WorldSaver, LockRenderer

from examples.pybullet.utils.pybullet_tools.panda_utils import get_edge_grasps


from common.scn import Scenario, load_ee




# Global
pandaNumDofs = 7
initialJointPoses = (0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0)

register(
    id='EdgePushing-v1',
    entry_point='env_vision:EdgePushingEnv',
    max_episode_steps=20,
)

class EdgePushingEnv(gym.Env):
    def __init__(self):
        super(EdgePushingEnv, self).__init__()
        
        # Action space
        # (x_1, y_1, x_2, y_2)
        self.action_space = spaces.Box(low=np.array([0, 0, 0], dtype=np.float32), high=np.array([1, 1, 1], dtype=np.float32))
        # self.action_segments_angle = 8
        # self.action_segments_length = 20
        # self.action_space = spaces.MultiDiscrete([self.action_segments_angle, self.action_segments_length])
        self.x_range = 0.5
        self.y_range = 0.5
        

        # State space
        self.workspace = np.array([[0.25, 0.75], [-0.25, 0.25]])
        self.reduce_factor = 1
        self.img_height = int(480/self.reduce_factor)
        self.img_width = int(640/self.reduce_factor)
        self.img_channel = 2
        self.observation_space = spaces.Box(low=0, high=2,
                                            # shape=(self.img_height, self.img_width), dtype=np.uint8)
                                            shape=(self.img_height, self.img_width), dtype=np.uint8)
                                            
        

        # Scene
        self.connect()

        # Camera
        self.camera_hight = 0.4 # This is the absolute hight 

        # Data
        # 
        self.episode_return = 0
        self.episode_length = 0
        self.total_steps = 0
        self.data = pd.DataFrame(columns=['episode_return', 'episode_length', 'total_steps'])

    ####################################################
    def connect(self):
        self.sim_id = connect(use_gui=True)

        self.scn = Scenario()
        self.scn.reset()

        # Robot
        self.robot = self.scn.add_ee_mobile(pose=((0,0,0.28), (0,0,0,1)))
        
        # Table
        # self.table = self.scn.add_low_table() # The height of the table surface is 0.3
        self.table = self.scn.add_edge()

        # Objects
        
        self.target, self.target_type = self.scn.add_plate_random()
 
        self.ee = load_ee(pos=(1,-1,1), mode='open')

        # Goal
        # self.goal_mark = self.scn.add_goal_mark()
        self.obj_info = {self.robot: 'robot', self.table: 'table', self.target: 'plate'}




    ####################################################

    def reset(self):

        print("\033[92m {}\033[00m" .format(f'\n New Episode'))

        # ----------------------------------------------
        # Reset Table
        quat = quaternion_from_euler(0, 0, np.random.uniform(low=-math.pi/36, high=math.pi/36))
        # quat = quaternion_from_euler(0, 0, np.random.normal(loc=0, scale=math.pi/12))
        pb.resetBasePositionAndOrientation(self.table, (0, 0, 0.075), quat)
        
        # ----------------------------------------------
        # Reset Objects
        pb.removeBody(self.target)
        self.target, self.target_type = self.scn.add_plate_random() # 0 cylinder, 1 box
        # self.target, self.target_type = self.scn.add_plate()
        pb.changeDynamics(self.target, -1, angularDamping=0.01, lateralFriction=1)
        initial_done = True
        while initial_done == True:
            pb.setRealTimeSimulation(1)
            x_obj = 0 #np.random.uniform(low = -0.05, high= 0.05)
            y_obj = np.random.uniform(low = -0.1, high = 0.1) #np.random.uniform(low=self.workspace[1][0] + 0.2, high=self.workspace[1][1] - 0.2)
            z_obj = 0.17
            quat = quaternion_from_euler(0, 0, np.random.uniform(low=0, high=2 * math.pi))
            pb.resetBasePositionAndOrientation(self.target, (x_obj, y_obj, z_obj), quat)
            self.initial_pose = pb.getBasePositionAndOrientation(self.target)
            _, initial_done = self.reward()
        
        # ----------------------------------------------
        # Reset Robot
        pb.resetBasePositionAndOrientation(self.robot, (0, 0, 1), (0,0,0,1))

        # ----------------------------------------------
        time.sleep(0.2)
        # pb.setRealTimeSimulation(0)
        self.previous_pose = pb.getBasePositionAndOrientation(self.target)
        self.initial_pose = pb.getBasePositionAndOrientation(self.target)

        # ----------------------------------------------
        observation = self.observe()
        
        # Data
        self.data.loc[len(self.data.index)] = [self.episode_return, self.episode_length, self.total_steps]
        self.data.to_csv('./training_data/training_data.csv', index=False)
        self.episode_return = 0
        self.episode_length = 0

        
        return observation


   
    ####################################################

    def step(self, action):

        
        print("\033[30m {}\033[00m" .format(f'\n {self.episode_length + 1} Step'))


        print(f' Action: {action}')

        # Action
        self.previous_pose = pb.getBasePositionAndOrientation(self.target)
        self.render(action)

        reward, done = self.reward()
        observation = self.observe()
        # plt.imshow(observation)
        # plt.savefig('./observation.png')


        # print(f' Observation: {observation}')
        # print(f' Reward: {reward}')
        # print(f' Done: {done}')
        pose = pb.getBasePositionAndOrientation(self.target)
        print(f' Pose: {pose}')


        # Data
        self.episode_return += reward
        self.episode_length += 1
        self.total_steps += 1

        # Done
        if self.episode_length >= 20:
            done = True

        # plt.imshow(observation)
        # plt.show()


        return observation, reward, done, {}


    ####################################################

    def observe(self):

        pos, orn = pb.getBasePositionAndOrientation(self.target)
        
        viewMatrix = pb.computeViewMatrix(
                    cameraEyePosition=[pos[0], pos[1], 0.35], # 0.35 when local observation; 0.55
                    cameraTargetPosition=[pos[0], pos[1], 0],
                    # cameraEyePosition=[self.initial_pose[0][0], self.initial_pose[0][1], 0.35],
                    # cameraTargetPosition=[self.initial_pose[0][0], self.initial_pose[0][1], 0], 
                    cameraUpVector=[1, 0, 0])
        

        near = 0.01
        far = 1
    
        projectionMatrix = pb.computeProjectionMatrixFOV(
                        fov= 74,
                        aspect= 640/480,
                        nearVal= near,
                        farVal=far)
        
        img = pb.getCameraImage(width=640, 
                                height=480,
                                viewMatrix=viewMatrix,
                                projectionMatrix=projectionMatrix, 
                                renderer=pb.ER_BULLET_HARDWARE_OPENGL)
        
        depth_img_raw = np.asarray(img[3], dtype=np.float32) 
        rgb_img = img[2]

        depth_map = far * near / (far - (far - near) * depth_img_raw)

        # for w in range(0, depth_img.shape[1]):
        #     for h in range(0, depth_img.shape[0]):
        #         depthImg = float(depth_img[h, w])
        #         depth = far * near / (far - (far - near) * depthImg)

        # depth_img = cv2.resize(depth_img, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
        

        # observation = depth_img

        return depth_map, rgb_img


########################################################
# Tools
########################################################

def take_a_photo():
    viewMatrix = pb.computeViewMatrix(
                # cameraEyePosition=[pos[0], pos[1], 0.4],
                # cameraTargetPosition=[pos[0], pos[1], 0],
                cameraEyePosition=[1, 0., 0.6],
                cameraTargetPosition=[0.5, 0., 0.15],
                cameraUpVector=[-1, 0, 0])
    near = 0.01
    far = 5
    projectionMatrix = pb.computeProjectionMatrixFOV(
                    fov= 86,
                    aspect= 640/480,
                    nearVal= near,
                    farVal=far)
    
    img = pb.getCameraImage(width=640, 
                            height=480,
                            viewMatrix=viewMatrix,
                            projectionMatrix=projectionMatrix, 
                            renderer=pb.ER_BULLET_HARDWARE_OPENGL)
    image_bgr = cv2.cvtColor(img[2], cv2.COLOR_RGB2BGR)
    return image_bgr


def reward_visualization(env):
    obj_pose = pb.getBasePositionAndOrientation(env.target)

    # Can be grasped
    obstacles = [env.table, env.robot, 0]
    grasps = get_edge_grasps(env.target, target_type = env.target_type)
    for grasp in grasps:

        ee = load_ee(pos=(1,-1,1), mode='open')
        grasp_pose = multiply(obj_pose, invert(grasp)) 

        pb.setRealTimeSimulation(0)
        pb.resetBasePositionAndOrientation(ee, grasp_pose[0], grasp_pose[1])
        pb.performCollisionDetection()
        
        collision = False
        for i in obstacles:
            contact_points = pb.getContactPoints(ee, i)
            if len(contact_points) > 0:
                collision = True
                pb.changeVisualShape(ee, 0, rgbaColor=[1,0,0,0.5])
                pb.changeVisualShape(ee, 1, rgbaColor=[1,0,0,0.5])
                pb.changeVisualShape(ee, 2, rgbaColor=[1,0,0,0.5])
                # pb.changeVisualShape(ee, 0, rgbaColor=[1,0,0,1])
                break

        if collision == False:
            print(' Possible Grasp!')
            pb.changeVisualShape(ee, 0, rgbaColor=[0,1,0,0.5])
            pb.changeVisualShape(ee, 1, rgbaColor=[0,1,0,0.5])
            pb.changeVisualShape(ee, 2, rgbaColor=[0,1,0,0.5])

            reward = 10
            done = True
            # break

def angle_between(v1, v2):
    v1_u = v1 if np.linalg.norm(v1) == 0 else v1 / np.linalg.norm(v1)
    v2_u = v2 if np.linalg.norm(v2) == 0 else v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))    

def get_differences_matrix(A):
    # Define the kernel for the convolution operation
    kernel = np.array([[0, 0, 0],
                       [-1, 0, 1],
                       [0, 0, 0]])
    
    # Perform the convolution operation on matrix A with the defined kernel
    differences_matrix = convolve(A, kernel, mode='constant', cval=0.0)
    
    return np.abs(differences_matrix)
        
if __name__ == '__main__':
    # sim_id = connect(use_gui=True)

    env = EdgePushingEnv()
    # env.reset()


    quat = quaternion_from_euler(0, 0, 0) #np.random.uniform(low=0, high=2 * math.pi))
    pb.resetBasePositionAndOrientation(env.target, (0.5, 0.05, 0.165), quat)

    pb.setRealTimeSimulation(1)
    time.sleep(1.5)

    dep_map, rgb_img = env.observe()
    plt.imshow(dep_map)
    plt.show()

    # cv2.imwrite('./rgb_img.png', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)) 
    # dep_img = cv2.normalize(dep_img, None, 100, 200, cv2.NORM_MINMAX)
    # dep_img = dep_img.astype(np.uint8)
    # rgb_img = cv2.normalize(rgb_img, None, 0, 255, cv2.NORM_MINMAX)
    # rgb_img = rgb_img.astype(np.uint8)

    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dep_img)

    # Object layer
    # dep_img_o = copy.deepcopy(dep_img)
    # dep_img_o[dep_img > (min_val + 10)] = 200
    # edges_o = cv2.Canny(dep_img_o, 100, 110, apertureSize=3)
    # dep_img_o[edges_o == 255] = (255)
    
    # edge_positions = np.where(dep_img_o == 255)


    # dst = cv2.cornerHarris(dep_img_o,2,3,0.04)
    # corner_positions = np.where(dst > 0.05*dst.max())

    # for i in range(len(corner_positions[0])):
    #     cv2.circle(rgb_img, (corner_positions[1][i], corner_positions[0][i]), 10, (255, 0, 0), -1)

    # for i in range(10):
    #     np.random.seed(i)
    #     index = np.random.randint(0, len(edge_positions[0]))
    #     print(index)
    #     cv2.circle(dep_img_o, (edge_positions[1][index], edge_positions[0][index]), 10, (255, 0, 0), -1)
    

    # ##########################################

    # Table layer
    # dep_img_t = copy.deepcopy(dep_img)
    # dep_img_t[dep_img < (max_val - 10)] = 100

    diff = get_differences_matrix(dep_map)

    # edges_t = cv2.Canny(dep_img_t, 100, 110, apertureSize=3)
    # edges_t[edges_o == 255] = (0) # Eliminate the object edges
    # dep_img_t[edges_t == 255] = (255)

    # edge_positions = np.where(diff > 100) #np.where(dep_img_t == 255)

    # for i in range(len(edge_positions[0])):
    #     cv2.circle(dep_img_t, (edge_positions[1][i], edge_positions[0][i]), 10, (255, 0, 0), -1)
    
    img = np.concatenate((dep_map, diff), axis=1) 
    # cv2.imshow('result', img)
    plt.imshow(img)
    plt.show()
    # cv2.waitKey(0)

    input('next...')

    





















#######################################################################

    # Ploting for the paper

#######################################################################    
    # quat = quaternion_from_euler(0, 0, np.random.uniform(low=0, high=2 * math.pi))
    # pb.setRealTimeSimulation(1)


    
    # # Img 1
    # pb.resetBasePositionAndOrientation(env.target, (0.52, 0.03, 0.165), quat)
    # img_d = env.observe()
    # img_rgb = take_a_photo()
    # plt.imshow(img_d*(255/2))
    # plt.show()
    # cv2.imwrite('./training_data/edge_layer_1.png', img_d*(254/2))
    # # cv2.imwrite('./training_data/rgb_1.png', img_rgb)
    
    # # Img 2
    # pb.resetBasePositionAndOrientation(env.target, (0.52, 0.24, 0.165), quat)
    # img_d = env.observe()
    # img_rgb = take_a_photo()
    # cv2.imwrite('./training_data/edge_layer_2.png', img_d*(255/2))
    # # cv2.imwrite('./training_data/rgb_2.png', img_rgb)

    # # Img 3
    # pb.resetBasePositionAndOrientation(env.target, (0.52, 0.35, 0.015), quat)
    # img_d = env.observe()
    # img_rgb = take_a_photo()
    # cv2.imwrite('./training_data/edge_layer_3.png', img_d*(255/2))
    # # cv2.imwrite('./training_data/rgb_3.png', img_rgb)



#######################################################################
    # pb.disconnect()

    # env = EdgePushingEnv()

    # pb.setRealTimeSimulation(1)

    # # Img 1
    # # env.reset()
    # pb.resetBasePositionAndOrientation(env.target, (0.52, 0.03, 0.165), quat)
    # reward_visualization(env)
    # img_reward = take_a_photo()
    # cv2.imwrite('./training_data/reward_1.png', img_reward)


    # # Img 2
    # pb.disconnect()
    # env = EdgePushingEnv()
    # pb.setRealTimeSimulation(1)
    # pb.resetBasePositionAndOrientation(env.target, (0.52, 0.24, 0.165), quat)
    # reward_visualization(env)
    # img_reward = take_a_photo()
    # cv2.imwrite('./training_data/reward_2.png', img_reward, )


    # # Img 3
    # pb.disconnect()
    # env = EdgePushingEnv()
    # pb.setRealTimeSimulation(1)
    # pb.resetBasePositionAndOrientation(env.target, (0.52, 0.35, 0.015), quat)
    # reward_visualization(env)
    # img_reward = take_a_photo()
    # cv2.imwrite('./training_data/reward_3.png', img_reward, )



#######################################################################

    # Img 1
    # pb.resetBasePositionAndOrientation(env.target, (0.52, 0.03, 0.165), quat)
    # img_rgb = env.get_rect()
    # cv2.imwrite('./training_data/rect_1.png', img_rgb)
    
    # # Img 2
    # pb.resetBasePositionAndOrientation(env.target, (0.52, 0.23, 0.165), quat)
    # img_rgb = env.get_rect()
    # cv2.imwrite('./training_data/rect_2.png', img_rgb)

    # # Img 3
    # pb.resetBasePositionAndOrientation(env.target, (0.52, 0.35, 0.015), quat)
    # img_rgb = env.get_rect()
    # cv2.imwrite('./training_data/rect_3.png', img_rgb)



