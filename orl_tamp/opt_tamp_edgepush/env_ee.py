
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

class EdgePushEnv(gym.Env):
    def __init__(self):
        super(EdgePushEnv, self).__init__()
        
        # Action space
        # (x_1, y_1, x_2, y_2)
        self.action_space = spaces.Box(low=np.array([0, 0, 0], dtype=np.float64), high=np.array([1, 1, 1], dtype=np.float64))
        # self.action_segments_angle = 8
        # self.action_segments_length = 20
        # self.action_space = spaces.MultiDiscrete([self.action_segments_angle, self.action_segments_length])
        self.x_range = 0.5
        self.y_range = 0.5
        

        # State space
        self.workspace = np.array([[0.25, 0.75], [-0.25, 0.25]])
        self.reduce_factor = 5
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
        
        depth_img = cv2.medianBlur(img[3], 5)
        
        object_layer = np.zeros(shape=depth_img.shape)
        edge_layer = np.zeros(shape=depth_img.shape)
 
        for w in range(0, depth_img.shape[1]):
            for h in range(0, depth_img.shape[0]):
                depthImg = float(depth_img[h, w])
                depth = far * near / (far - (far - near) * depthImg)
                object_layer[h, w] = 2 if (depth < 0.19) else 0
                object_layer[h, w] = 1 if (depth > 0.19) and (depth < 0.25) else object_layer[h, w]
                # edge_layer[h, w] = 1 if (depth < 0.25) else 0


        object_layer = cv2.resize(object_layer, (self.img_width, self.img_height))
        
                
        # stride = (self.reduce_factor, self.reduce_factor)
        # object_layer = cv2.resize(object_layer, (0,0), fx=1/stride[0], fy=1/stride[1], interpolation=cv2.INTER_MAX)
        # edge_layer = cv2.resize(edge_layer, (self.img_width, self.img_height))

        # observation = np.array([object_layer, edge_layer], dtype=np.uint8)
        observation = np.array(object_layer, dtype=np.uint8)

        return observation

 

    ####################################################
    # Action 
    def render(self, action):

        pos, orn = pb.getBasePositionAndOrientation(self.target)

        x_0 = pos[0]
        y_0 = pos[1]

        r_start = 0.15
        r_end = 0.06
        l_max = 0.2

        theta_2_range = math.pi

    
        ##############################################
        theta_1 = action[0] * 2 * math.pi
        theta_2 = action[1] * theta_2_range
        l = action[2] * l_max

        x_1 = r_start * math.cos(theta_1) + x_0
        y_1 = r_start * math.sin(theta_1) + y_0

        # x_2 = r_end * math.cos(theta_2) + x_0
        # y_2 = r_end * math.sin(theta_2) + y_0

        # x_2 = x_0
        # y_2 = y_0

        # theta = np.arctan2(y_2 - y_1, x_2 - x_1)
        theta = theta_1 + theta_2 + (2 * math.pi-theta_2_range)/2
        theta = theta % (2 * math.pi)

        x_3 = x_1 + l * math.cos(theta)
        y_3 = y_1 + l * math.sin(theta)

        start_point = np.array([x_1, y_1])
        end_point = np.array([x_3, y_3])
        
        # ------------------------------------------------------
        # Yaw
        yaw = angle_between([1,0], np.subtract(end_point, start_point))
        if start_point[1] > end_point[1]:
            yaw = -yaw
        quat = quaternion_from_euler(0, 0, yaw)    
        
        # ------------------------------------------------------
        d = np.linalg.norm(start_point - end_point) # euclidean_distance(start_point, end_point)


        # ------------------------------------------------------
        # Collision Checking
        obstacles = [self.target]
        pb.setRealTimeSimulation(0)
        pb.resetBasePositionAndOrientation(self.robot, (x_1, y_1, 0.27), quat)
        pb.performCollisionDetection()
        for i in obstacles:
            contact_points = pb.getContactPoints(self.robot, i)
            if len(contact_points) > 0:
                collision = True
                print(' Start pose collision.') 
                pb.resetBasePositionAndOrientation(self.robot, (0, 0, 1), (0,0,0,1))
                pb.resetJointState(self.robot, 0, 0)
                return
            
        # ------------------------------------------------------
        # Execute
        
        pb.resetBasePositionAndOrientation(self.robot, (x_1, y_1, 0.27), quat)

        pb.setRealTimeSimulation(1)

        velocity = 0.4
      
        pb.setJointMotorControl2(bodyUniqueId=self.robot,
                                    jointIndex=0,
                                    controlMode=pb.POSITION_CONTROL,
                                    targetPosition=d,
                                    maxVelocity=velocity,
                                    force=500)
            # pb.resetBasePositionAndOrientation(self.robot, (segments[i], 0, 0.27), quat)
            # time.sleep(0.02)

        while True:
            d_now = pb.getJointState(self.robot, 0)[0]
            if abs(d_now - d) < 0.01:
                break
            time.sleep(0.02)

        pb.setJointMotorControl2(bodyUniqueId=self.robot,
                                    jointIndex=0,
                                    controlMode=pb.POSITION_CONTROL,
                                    targetPosition=0,
                                    maxVelocity=velocity,
                                    force=500)
        while True:
            d_now = pb.getJointState(self.robot, 0)[0]
            if abs(d_now - 0) < 0.01:
                break
            time.sleep(0.02)
        pb.setRealTimeSimulation(0)

        # ------------------------------------------------------
        # Reset
        pb.resetJointState(self.robot, 0, 0) 
        pb.resetBasePositionAndOrientation(self.robot, (0, 0, 1), (0,0,0,1))
        



    ####################################################
        

    def reward(self):
        done = False
        reward = -5

        obj_pose = pb.getBasePositionAndOrientation(self.target)
        travel = np.linalg.norm(np.array(obj_pose[0]) - np.array(self.initial_pose[0]))
        
        
        # ------------------------------------------------------
        # Off the table
        if obj_pose[0][2] < 0.15:
            print(' Object is on the ground.')
            reward -= 0 #travel * 20
            done = True
            # pb.resetBasePositionAndOrientation(self.target, self.previous_pose[0], self.previous_pose[1])
            return reward, done
        

        # ------------------------------------------------------
        # Can be grasped
        obstacles = [self.table, self.robot]
        grasps = get_edge_grasps(self.target, body_type = self.target_type)
        for grasp in grasps:
            grasp_pose = multiply(obj_pose, invert(grasp)) 

            pb.setRealTimeSimulation(0)
            pb.resetBasePositionAndOrientation(self.ee, grasp_pose[0], grasp_pose[1])
            pb.performCollisionDetection()
            
            collision = False
            for i in obstacles:
                contact_points = pb.getContactPoints(self.ee, i)
                if len(contact_points) > 0:
                    collision = True
                    break
            if collision == False:
                break
        pb.resetBasePositionAndOrientation(self.ee, [1,-1,1], [0,0,0,1])
        # ------------------------------------------------------

        if collision == False:
            
            # reward -= 10 * travel

            print(' Goal Reached!')
            reward = 10
            done = True   
                
        
        return reward, done
    
    ###################################################

    def render_goal(self, goal):
        goal = goal.reshape((-1,))
        position = [goal[0], goal[1], 0.0005] 
        orn = [0,0,0,1]#quaternion_from_euler(0, 0, goal[2])
        # pb.resetBasePositionAndOrientation(self.scn.cube_shadow, pos, orn)
        pb.resetBasePositionAndOrientation(self.scn.goal_mark, position, orn)

    ##################################################

  
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

        
if __name__ == '__main__':
    # sim_id = connect(use_gui=True)

    env = EdgePushEnv()
    # env.reset()


    quat = quaternion_from_euler(0, 0, 0) #np.random.uniform(low=0, high=2 * math.pi))

 
    input('next...')
    pb.resetBasePositionAndOrientation(env.target, (0.5, 0.0, 0.165), (0,0,0,1))
    env.render([0.75, 0.5, 1])
    input('next...')

    pb.resetBasePositionAndOrientation(env.target, (0.5, 0.0, 0.165), (0,0,0,1))
    env.render([0, 0.75, 1])
    input('next...')
    # for i in range(10):
    #     x_obj = np.random.uniform(low = -0.05, high= 0.05)
    #     y_obj = np.random.uniform(low = -0.05, high = 0.05) #np.random.uniform(low=self.workspace[1][0] + 0.2, high=self.workspace[1][1] - 0.2)
    #     z_obj = 0.17
    #     quat = quaternion_from_euler(0, 0, np.random.uniform(low=0, high=2 * math.pi))

    #     pb.resetBasePositionAndOrientation(env.table, (0, 0, 0.075), quat)
    #     pb.resetBasePositionAndOrientation(env.target, (x_obj, y_obj, z_obj), quat)
        

    #     pb.setRealTimeSimulation(1)
    #     time.sleep(1)

    #     env.render([0.75, .5])

    #     obs = env.observe()

    #     plt.imshow(obs)
    #     plt.show()
 

    #     input('next...')

    





















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



