
#! /venv3.8/bin/python
from email.policy import default
import os, sys
from tokenize import Double
import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np
import math
import pybullet as p
import time
import colorama
from colorama import Fore
import pandas as pd

# Import Searching Dict
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../pddlstream/')
sys.path.insert(0, file_path)

from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion
from examples.pybullet.utils.pybullet_tools.utils import connect, wait_for_user

from control import Control
sys.path.insert(0, file_path + '/../')
from common.scn import Scenario
from motion_planner import CartesianPlanner
# from solver.solver import Solver


# Global
pandaNumDofs = 7
initialJointPoses = (0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0)

register(
    id='Retrieve-v0',
    entry_point='env:RetrieveEnv',
    max_episode_steps=20,
)

class RetrieveEnv(gym.Env):
    def __init__(self):
        super(RetrieveEnv, self).__init__()

        # Mode
        self.mode = 'train'
        
        # Action space
        # self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float64), high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float64))
        self.action_space = spaces.Box(low=np.array([-1, -1, -1], dtype=np.float64), high=np.array([1, 1, 1], dtype=np.float64))

        # State space
        # [cup, bae]
        self.observation_space = spaces.Box(low=np.array([0, -1.2, -0.5, -1.2, -3.14]), 
                                            high=np.array([1.2, 1.2, 1.2, 1.2, 3.14]),
                                            dtype=np.float64)

        # Action Tools
        self.cartesian_planner = CartesianPlanner()
        self.control = Control()

        # Goal
        self.goal = np.array([0.5, 0.3, 0.0])

        # Debug
        self.total_steps = 0
        self.connect()

        # Data
        self.episodic_return = 0
        self.episode_length = 0
        self.total_steps = 0
        self.data = pd.DataFrame(columns=['episodic_return', 'episode_length', 'total_steps'])

    ####################################################
    def connect(self, use_gui=True):
        self.sim_id = connect(use_gui=True)
        self.scn = Scenario()
        # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        # p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

        # Objects
        self.scn.reset()
        self.robot = self.scn.add_robot(pose=((0,0,0.0), (0,0,0,1)))
        self.control.finger_open(self.robot)

        self.table = self.scn.add_low_table()

        self.scn.add_cube()
        self.scn.add_bar()

        self.target = self.scn.cube
        self.bar = self.scn.bar

        self.goal_mark = self.scn.add_goal_mark()
        
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
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, initialJointPoses[i],force=5 * 240.)
        
        # ------------------------------
        # Reset Objects
        # 
        r = np.random.uniform(low=0.8, high=1.1)
        theta = np.random.uniform(low=-math.pi/6, high=math.pi/6)
        x = np.random.uniform(low=0.75, high=1.1) #r*math.cos(theta)
        y = np.random.uniform(low=-0.2, high=0.2) #r*math.sin(theta)
        f_coeff = 20 #np.random.uniform(low=0.4, high=1.0)
        
        z = 0.025 +0.15
        yaw = np.random.uniform(low=-math.pi, high=math.pi)
        orn = quaternion_from_euler(0,0,yaw)
        p.resetBasePositionAndOrientation(self.target, [x, y, z], orn)
        # Change friction
        p.changeDynamics(self.target, -1, lateralFriction=f_coeff)

        # ------------------------------
        # Reset Hook
        # If hook is not reset, then the training will stuck in dead-end  
        r_hook = np.random.uniform(low=0.4, high=0.7)
        theta_hook = np.random.uniform(low=-math.pi/6, high=math.pi/6)
        x_hook = np.random.uniform(low=0.35, high=0.6) #r_hook*math.cos(theta_hook)
        y_hook = np.random.uniform(low=-0.2, high=0.2) #r_hook*math.sin(theta_hook)
       
        yaw_hook = np.random.uniform(low=-math.pi/4, high=math.pi/4)
        orn_hook = quaternion_from_euler(0,0,yaw_hook)
        orn_hook = quaternion_from_euler(0,0,0)
        p.resetBasePositionAndOrientation(self.bar, (x_hook, y_hook, 0.16), orn_hook)

        # ------------------------------
        # Reset Goal
        r_goal = np.random.uniform(low=0.4, high=0.6)
        theta_goal = np.random.uniform(low=-math.pi/4, high=math.pi/4)
        x_goal = r_goal*math.cos(theta_goal)
        y_goal = r_goal*math.sin(theta_goal)
        p.resetBasePositionAndOrientation(self.goal_mark, (x_goal, y_goal, 0.151), (0,0,0,1))
        self.goal = np.array([x_goal, y_goal, 0.151])
        # self.goal = np.array([np.random.uniform(low=0.3, high=0.6), np.random.uniform(low=-0.4, high=0.4), 0.0])
        
        # ------------------------------
        p.setRealTimeSimulation(1)
        self.observation = self.observe()

        self.pre_dis = np.linalg.norm(self.observation[:2] - self.goal[:2])

        # Data
        self.data.loc[len(self.data.index)] = [self.episodic_return, self.episode_length, self.total_steps]
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.data.to_csv(file_path+'/training_data/training_data.csv', index=False)
        self.episodic_return = 0
        self.episode_length = 0

        return self.observation
    
    ####################################################
    def reset_eval(self):
        p.setRealTimeSimulation(1)
        print("\033[92m {}\033[00m" .format(f'\n New Episode'))

        # Reset Robot
        for i in range(pandaNumDofs):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, initialJointPoses[i],force=5 * 240.)
        
        # ------------------------------
        # Reset Objects
        x = np.random.uniform(low=0.0, high=1)
        y = np.random.uniform(low=-1, high=1)
        f_coeff = 1 #np.random.uniform(low=0.4, high=1.0)
        
        z = 0.025
        yaw = np.random.uniform(low=-math.pi, high=math.pi)
        orn = quaternion_from_euler(0,0,yaw)
        p.resetBasePositionAndOrientation(self.target, [x, y, z], orn)
        # Change friction
        
        p.changeDynamics(self.target, -1, lateralFriction=f_coeff)

        # ------------------------------
        # Reset Hook
        x_hook = 0.5
        y_hook = 0.0
        yaw_hook = np.random.uniform(low=-math.pi/4, high=math.pi/4)
        orn_hook = quaternion_from_euler(0,0,yaw_hook)
        orn_hook = quaternion_from_euler(0,0,0)
        p.resetBasePositionAndOrientation(self.bar, (x_hook, y_hook, 0), orn_hook)

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

        pos, orn = p.getBasePositionAndOrientation(self.target)
        rpy = p.getEulerFromQuaternion(orn)
        total.append(pos[0])
        total.append(pos[1])

        pos, orn = p.getBasePositionAndOrientation(self.bar)
        rpy = p.getEulerFromQuaternion(orn)
        total.append(pos[0])
        total.append(pos[1])
        total.append(rpy[-1])

        obs = np.array(total)
        
        return obs

    ####################################################

    def render(self, action):

        # Initial position
        # theta_0 = action[0] * math.pi/2
        # r_0 = 0.55 + action[1] * 0.15
        # yaw_0 = action[2] * math.pi/2
        # x_0 = r_0 * math.cos(theta_0)
        # y_0 = r_0 * math.sin(theta_0)
        # pos_0 = [x_0, y_0, 0.02]
        # orn_0 = quaternion_from_euler(0,0,yaw_0)
        # p.resetBasePositionAndOrientation(self.bar, pos_0, orn_0)

        # ------------------------------

        # Goal position
        # d_theta = action[0] * math.pi/2
        # d_r = 0.55 + action[1] * 0.15
        d_yaw = action[2] * math.pi/2

        d_x = 0.5 + 0.2 * action[0] #d_r * math.cos(d_theta)
        d_y = 0.4 * action[1] #d_r * math.sin(d_theta)

        # ------------------------------

        # Go approach
        info = p.getBasePositionAndOrientation(self.bar)
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
        info = p.getBasePositionAndOrientation(self.bar)
        ee_pos2 = list(info[0])
        ee_pos2[-1] = ee_pos1[-1] - 0.04
        ee_orn2 = ee_orn1
        grasp_pose = (ee_pos2, ee_orn2)
        end_pose_list = [approach_pose, grasp_pose]
        joint_states = p.getJointStates(self.robot, list(range(pandaNumDofs)))# Given by Holding
        start_conf = []
        for i in joint_states:
            start_conf.append(i[0])
        success, trajectory = self.cartesian_planner.plan_multipose(start_conf, end_pose_list)
        for waypoint in trajectory:
            time.sleep(0.01) # Moving speed
            finger_state = p.getLinkState(self.robot, 11)[4]
            self.control.go_joint_space(self.robot, waypoint)
        # self.control.finger_open(self.robot)
        self.control.finger_close(self.robot)
        # ------------------------------


        # Slide
        # observation = self.observe()
        x = d_x
        y = d_y
        yaw_3 = d_yaw #ee_yaw1 + d_yaw 

        ee_orn3 = quaternion_from_euler(math.pi, 0, yaw_3)
        ee_pos3 = np.array([x, y, ee_pos2[-1]+0.03]).astype(float)
        goal_pose = (ee_pos3, ee_orn3)

        # Plan & execution
        end_pose_list = [goal_pose]
        joint_states = p.getJointStates(self.robot, list(range(pandaNumDofs)))# Given by Holding
        start_conf = []
        for i in joint_states:
            start_conf.append(i[0])
        success, trajectory = self.cartesian_planner.plan_multipose(start_conf, end_pose_list)
        for waypoint in trajectory:
            time.sleep(0.01) # Moving speed
            finger_state = p.getLinkState(self.robot, 11)[4]
            # if finger_state[-1] <= 0.05:
            self.control.finger_close(self.robot)
            self.control.go_joint_space(self.robot, waypoint)
        self.control.finger_open(self.robot)
        

        

        # Up
        if success:
            ee_pos4 = ee_pos3
            ee_pos4[-1] = ee_pos4[-1] + 0.1
            ee_orn4 = ee_orn3
            up_pose = (ee_pos4, ee_orn4)
            info = p.getBasePositionAndOrientation(self.bar)
            end_pose_list = [up_pose]
            joint_states = p.getJointStates(self.robot, list(range(pandaNumDofs)))# Given by Holding
            start_conf = []
            for i in joint_states:
                start_conf.append(i[0])
            _, trajectory = self.cartesian_planner.plan_multipose(start_conf, end_pose_list)
            for waypoint in trajectory:
                time.sleep(0.01) # Moving speed
                finger_state = p.getLinkState(self.robot, 11)[4]
                self.control.go_joint_space(self.robot, waypoint)
                self.control.finger_open(self.robot)



        # Reset
        joint_states = p.getJointStates(self.robot, list(range(pandaNumDofs)))# Given by Holding
        start_conf = []
        for i in joint_states:
            start_conf.append(i[0])
        default_config = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
        trajectory = self.cartesian_planner.plan_joint(start_conf, default_config)
        for waypoint in trajectory:
            time.sleep(0.02) # Moving speed
            self.control.go_joint_space(self.robot, waypoint)




    ####################################################


    def reward(self, obs):
        done = False
        reward = 0
        
        dis = np.linalg.norm(obs[:2] - np.array([0,0])) #self.goal[:2])
        
        hook_pose = p.getBasePositionAndOrientation(self.bar)
        object_pose = p.getBasePositionAndOrientation(self.target)


        # if hook_pose[0][-1] < 0.15:
        #     return -5, True
        
        # if object_pose[0][-1] < 0.15:
        #     return -5, True


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
        # p.resetBasePositionAndOrientation(self.scn.cube_shadow, pos, orn)
        p.resetBasePositionAndOrientation(self.scn.goal_mark, position, orn)

    
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


            


        
if __name__ == '__main__':
    # sim_id = connect(use_gui=True)

    env = HookingEnv()

    p.setRealTimeSimulation(1)
    wait_for_user()

    env.reset()
    wait_for_user()
    
    
    action = np.array((-0.7,1,1))
    env.render(action)

    wait_for_user()

    action = np.array((0.7,1,1))
    env.render(action)

    # observation_space = env.observation_space

    # print(observation_space.sample())

    wait_for_user()





