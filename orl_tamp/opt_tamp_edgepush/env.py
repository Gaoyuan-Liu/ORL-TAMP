
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
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Import Searching Dict
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../pddlstream/')

sys.path.insert(0, file_path + '/../')
from utils.scn import add_ground, add_robot, add_low_table, add_plate, load_ee, add_plate_random, add_table
from utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion
from utils.pybullet_tools.utils import connect, multiply, invert, WorldSaver, LockRenderer
from utils.pybullet_tools.panda_utils import get_edge_grasps
# from solver.solver import Solver
sys.path.insert(0, file_path)
from pushing import Pushing


# Global
pandaNumDofs = 7
initialJointPoses = (0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0)

register(
    id='EdgePush-v1',
    entry_point='env_vision:EdgePushEnv',
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
        self.reduce_factor = 10
        self.img_height = int(480/self.reduce_factor)
        self.img_width = int(640/self.reduce_factor)
        self.observation_space = spaces.Box(low=0, high=2,
                                            shape=(self.img_height, self.img_width), dtype=np.uint8)
        # self.observation_space = spaces.Box(low=np.array([-1, -1, -1, -1, 0, 0]), high=np.array([2, 1, 2, 1, 1, 1]), dtype=np.float64)
        # self.observation_space = spaces.MultiDiscrete(np.array([[3]*WIDTH] * HEIGHT))

        # Scene
        self.connect()

        # Camera
        self.camera_hight = 0.4 # This is the absolute hight 

        # Pusher
        self.pushing = Pushing()

        # Data
        self.episode_return = 0
        self.episode_length = 0
        self.total_steps = 0
        self.data = pd.DataFrame(columns=['episode_return', 'episode_length', 'total_steps'])

    ####################################################
    def connect(self):
        self.sim_id = connect(use_gui=True)

        self.scn = {} 
        
        # Ground
        ground = add_ground()
        self.scn['ground'] = ground

        # Robot
        robot = add_robot(pose=((0,0,0), (0,0,0,1)))
        self.scn['robot'] = robot
        
        # Table
        self.table = add_low_table() # The height of the table surface is 0.3
        self.scn['table'] = self.table

        # Objects
        dish, type = add_plate()
        self.target = dish
        self.scn['dish'] = dish
        self.body_type = type
 
        self.reward_ee = load_ee(pos=(1,-1,1), mode='open')

        self.obstacles = [self.scn['table'], self.scn['dish']]




    ####################################################

    def reset(self):

        print("\033[92m {}\033[00m" .format(f'\n New Episode'))

        # ----------------------------------------------
        # Reset Table
        quat = quaternion_from_euler(0, 0, np.random.uniform(low=-math.pi/12, high=math.pi/12))
        # quat = quaternion_from_euler(0, 0, np.random.normal(loc=0, scale=math.pi/12))
    
        
        # ----------------------------------------------
        # Reset Objects
        pb.removeBody(self.target)
        self.target, self.body_type = add_plate_random() # 0 cylinder, 1 box
        pb.changeDynamics(self.target, -1, lateralFriction=2, angularDamping=0.01)
        initial_done = True
        while initial_done == True:
            pb.setRealTimeSimulation(1)
            x_obj = np.random.uniform(low = -0.05, high= 0.05)
            y_obj = np.random.uniform(low = -0.05, high = 0.05) #np.random.uniform(low=self.workspace[1][0] + 0.2, high=self.workspace[1][1] - 0.2)
            z_obj = 0.17
            quat = quaternion_from_euler(0, 0, np.random.uniform(low=0, high=2 * math.pi))
            pb.resetBasePositionAndOrientation(self.target, (x_obj+0.5, y_obj+0.15, z_obj), quat)
            self.initial_pose = pb.getBasePositionAndOrientation(self.target)
            print(f' Initial Pose: {self.initial_pose}')
            _, initial_done = self.reward()
        
        # ----------------------------------------------
        # Reset Robot


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
        self.render(action)

        observation = self.observe()

        reward, done = self.reward()


        # print(f' Observation: {observation}')
        print(f' Reward: {reward}')
        print(f' Done: {done}')

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
                    # cameraEyePosition=[0.5, 0, 0.55], # 0.35 when local observation; 0.55
                    # cameraTargetPosition=[0.5, 0, 0],
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
        
        object_layer = np.zeros(shape=img[3].shape)
 
        for w in range(0, img[0]):
            for h in range(0, img[1]):
                depthImg = float(img[3][h, w])
                depth = far * near / (far - (far - near) * depthImg)
                object_layer[h, w] = 2 if (depth < 0.19) else 0
                object_layer[h, w] = 1 if (depth > 0.19) and (depth < 0.25) else object_layer[h, w]

        object_layer = cv2.resize(object_layer, (self.img_width, self.img_height))

        observation = np.array(object_layer, dtype=np.uint8)

        return observation


    ####################################################


    def render(self, action):

        pos, orn = pb.getBasePositionAndOrientation(self.target)

        x_0 = pos[0]
        y_0 = pos[1]

        r_start = 0.15
        r_end = 0.06
        l_max = 0.2

        theta_2_range = math.pi

        theta_1 = action[0] * 2 * math.pi
        theta_2 = action[1] * theta_2_range
        l = action[2] * l_max

        x_1 = r_start * math.cos(theta_1) + x_0
        y_1 = r_start * math.sin(theta_1) + y_0

        theta = theta_1 + theta_2 + (2 * math.pi-theta_2_range)/2
        theta = theta % (2 * math.pi)

        x_2 = x_1 + l * math.cos(theta)
        y_2 = y_1 + l * math.sin(theta)

        start_point = np.array([x_1, y_1])
        end_point = np.array([x_2, y_2])


        # ----------------------------------------------
        self.pushing.push(self.scn['robot'], start_point, end_point, self.obstacles)



    ####################################################
        

    def reward(self):
        done = False
        reward = -1

        obj_pose = pb.getBasePositionAndOrientation(self.target)
        # travel = np.linalg.norm(np.array(obj_pose[0]) - np.array(self.initial_pose[0]))
        
        
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
        grasps = get_edge_grasps(self.target, body_type = self.body_type)
        for grasp in grasps:
            # input('press...')
            grasp_pose = multiply(obj_pose, invert(grasp)) 

            pb.setRealTimeSimulation(0)
            pb.resetBasePositionAndOrientation(self.reward_ee, grasp_pose[0], grasp_pose[1])
            pb.performCollisionDetection()
            
            collision = False
            for i in self.obstacles:
                contact_points = pb.getContactPoints(self.reward_ee, i)
                if len(contact_points) > 0:
                    collision = True
                    break
            if collision == False:
                break
        pb.resetBasePositionAndOrientation(self.reward_ee, [1,-1,1], [0,0,0,1])
        # ------------------------------------------------------

        if collision == False:
            
            # reward -= 20 * travel

            print(' Goal Reached!')
            reward += 10
            done = True   
                
        return reward, done
    


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
    obj_pose = pb.getBasePositionAndOrientation(env.dish)

    # Can be grasped
    obstacles = [env.table, env.robot, 0]
    grasps = get_edge_grasps(env.dish, body_type = env.body_type)
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

    # env = EdgePushEnv()
    # env.reset()

    # quat = quaternion_from_euler(0, 0, np.random.uniform(low=0, high=2 * math.pi))
    


    # for i in range(10):
    #     action = np.random.uniform(low=np.array([0,0,0]), high=np.array([1,1,1]))
    #     env.render(action)
    #     input('next...')
        

    # pb.resetBasePositionAndOrientation(env.dish, (0.5, 0.0, 0.165), (0,0,0,1))
    # env.render([0,0])
    # input('next...')

    # pb.resetBasePositionAndOrientation(env.dish, (0.5, 0.0, 0.165), (0,0,0,1))
    # env.render([0,0.5])
    # input('next...')

    # pb.resetBasePositionAndOrientation(env.dish, (0.5, 0.0, 0.165), (0,0,0,1))
    # env.render([0,1])
    # input('next...')





















#######################################################################

    # Ploting for the paper

#######################################################################    
    quat = quaternion_from_euler(0, 0, np.random.uniform(low=0, high=2 * math.pi))
    pb.setRealTimeSimulation(1)


    
    # Img 1
    # pb.resetBasePositionAndOrientation(env.dish, (0.52, 0.03, 0.165), quat)
    # img_d = env.observe()
    # img_rgb = take_a_photo()
    # plt.imshow(img_d)
    # plt.show()
    # cv2.imwrite('./training_data/edge_layer_1.png', img_d*(254/2))
    # plt.savefig('./training_data/edge_layer_1.png', bbox_inches='tight')
    # cv2.imwrite('./training_data/rgb_1.png', img_rgb)
    
    # Img 2
    # pb.resetBasePositionAndOrientation(env.dish, (0.52, 0.24, 0.165), quat)
    # img_d = env.observe()
    # img_rgb = take_a_photo()
    # cv2.imwrite('./training_data/edge_layer_2.png', img_d*(255/2))
    # cv2.imwrite('./training_data/rgb_2.png', img_rgb)

    # Img 3
    # pb.resetBasePositionAndOrientation(env.dish, (0.52, 0.35, 0.015), quat)
    # img_d = env.observe()
    # img_rgb = take_a_photo()
    # cv2.imwrite('./training_data/edge_layer_3.png', img_d*(255/2))
    # cv2.imwrite('./training_data/rgb_3.png', img_rgb)



#######################################################################
    # pb.disconnect()

    # env = EdgePushingEnv()

    # pb.setRealTimeSimulation(1)

    # # Img 1
    # # env.reset()
    # pb.resetBasePositionAndOrientation(env.dish, (0.52, 0.03, 0.165), quat)
    # reward_visualization(env)
    # img_reward = take_a_photo()
    # cv2.imwrite('./training_data/reward_1.png', img_reward)


    # # Img 2
    # pb.disconnect()
    # env = EdgePushingEnv()
    # pb.setRealTimeSimulation(1)
    # pb.resetBasePositionAndOrientation(env.dish, (0.52, 0.24, 0.165), quat)
    # reward_visualization(env)
    # img_reward = take_a_photo()
    # cv2.imwrite('./training_data/reward_2.png', img_reward, )


    # # Img 3
    # pb.disconnect()
    # env = EdgePushingEnv()
    # pb.setRealTimeSimulation(1)
    # pb.resetBasePositionAndOrientation(env.dish, (0.52, 0.35, 0.015), quat)
    # reward_visualization(env)
    # img_reward = take_a_photo()
    # cv2.imwrite('./training_data/reward_3.png', img_reward, )



#######################################################################

    # Img 1
    # pb.resetBasePositionAndOrientation(env.dish, (0.52, 0.03, 0.165), quat)
    # img_rgb = env.get_rect()
    # cv2.imwrite('./training_data/rect_1.png', img_rgb)
    
    # # Img 2
    # pb.resetBasePositionAndOrientation(env.dish, (0.52, 0.23, 0.165), quat)
    # img_rgb = env.get_rect()
    # cv2.imwrite('./training_data/rect_2.png', img_rgb)

    # # Img 3
    # pb.resetBasePositionAndOrientation(env.dish, (0.52, 0.35, 0.015), quat)
    # img_rgb = env.get_rect()
    # cv2.imwrite('./training_data/rect_3.png', img_rgb)



