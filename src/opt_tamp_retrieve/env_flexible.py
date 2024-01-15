
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
    entry_point='env_flexible:RetrieveEnv',
    max_episode_steps=20,
)

class RetrieveEnv(gym.Env):
    def __init__(self):
        super(RetrieveEnv, self).__init__()

        # Mode
        self.mode = 'train'
        
        # Action space
        # self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float64), high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float64))
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float64), high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float64))

        # State space
        # [cup, hook]
        # self.observation_space = spaces.Box(low=np.array([0, -1.2, -0.5, -1.2, -3.14]), 
        #                                     high=np.array([1.2, 1.2, 1.2, 1.2, 3.14]),
        #                                     dtype=np.float64)
        
        self.observation_space = spaces.Box(low=np.array([0, -1.2]), 
                                            high=np.array([1.2, 1.2]),
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
        #pb.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
        #pb.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        #pb.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        #pb.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        #pb.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

        # Objects
        self.scn.reset()
        self.robot = self.scn.add_robot(pose=((0,0,0.0), (0,0,0,1)))
        self.control.finger_open(self.robot)

        self.table = self.scn.add_low_table()

        self.target = self.scn.add_cube_random()
        self.tool = self.scn.add_bar()
      

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
        # Reset Target
        pb.removeBody(self.target)
        self.target = self.scn.add_cube_random()
        # r = np.random.uniform(low=0.8, high=1.1)
        # theta = np.random.uniform(low=-math.pi/6, high=math.pi/6)
        x = np.random.uniform(low=0.8, high=1.1) #r*math.cos(theta)
        y = np.random.uniform(low=-0.2, high=0.2) #r*math.sin(theta)
        f_coeff = 0.8 #np.random.uniform(low=0.4, high=1.0)
        
        z = 0.025 + 0.15
        yaw = np.random.uniform(low=-math.pi, high=math.pi)
        orn = quaternion_from_euler(0,0,yaw)
        pb.resetBasePositionAndOrientation(self.target, [x, y, z], orn)
        # Change friction
        
        pb.changeDynamics(self.target, -1, lateralFriction=f_coeff)


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
        pb.resetBasePositionAndOrientation(self.tool, (x_hook, y_hook, 0.17), orn_hook)

        # ------------------------------
        # Reset Goal
        # r_goal = np.random.uniform(low=0.4, high=0.6)
        # theta_goal = np.random.uniform(low=-math.pi/4, high=math.pi/4)
        x_goal = np.random.uniform(low=0.3, high=0.6) # r_goal*math.cos(theta_goal)
        y_goal = np.random.uniform(low=-0.2, high=0.2) # r_goal*math.sin(theta_goal)
        # pb.resetBasePositionAndOrientation(self.goal_mark, (x_goal, y_goal, 0.151), (0,0,0,1))
        self.goal = np.array([x_goal, y_goal, 0.151])
        # self.goal = np.array([np.random.uniform(low=0.3, high=0.6), np.random.uniform(low=-0.4, high=0.4), 0.0])
        
        # ------------------------------
        pb.setRealTimeSimulation(1)
        observation = self.observe()

        self.pre_dis = np.linalg.norm(observation[:2] - self.goal[:2])

        # Data
        self.data.loc[len(self.data.index)] = [self.episodic_return, self.episode_length, self.total_steps]
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.data.to_csv(file_path+'/training_data/training_data.csv', index=False)
        self.episodic_return = 0
        self.episode_length = 0

        return observation
    
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
        pb.resetBasePositionAndOrientation(self.target, [x, y, z], orn)
        # Change friction
        
        pb.changeDynamics(self.target, -1, lateralFriction=f_coeff)

        # ------------------------------
        # Reset Hook
        x_hook = 0.5
        y_hook = 0.0
        yaw_hook = np.random.uniform(low=-math.pi/4, high=math.pi/4)
        orn_hook = quaternion_from_euler(0,0,yaw_hook)
        orn_hook = quaternion_from_euler(0,0,0)
        pb.resetBasePositionAndOrientation(self.tool, (x_hook, y_hook, 0), orn_hook)

        # ------------------------------
        # Reset Goal
        self.goal = np.array([np.random.uniform(low=0.3, high=0.6), np.random.uniform(low=-0.4, high=0.4), 0.0])
        observation = self.observe()
        self.pre_dis = np.linalg.norm(observation[:2] - self.goal[:2])

        self.episodic_return = 0
        self.episode_length = 0
        self.total_steps = 0

        return observation


    ####################################################

    def step(self, action):

        print("\033[30m {}\033[00m" .format(f'\n {self.episode_length + 1} Step'))

        self.render(action)
        observation = self.observe()
        print(f' observation = {observation}')

        reward, done = self.reward(observation)

        # Data
        self.episodic_return += reward
        self.episode_length += 1
        self.total_steps += 1
        
        # Done
        if self.episode_length >= 20:
            done = True

        return observation, reward, done, {}


    ####################################################

    def observe(self):
        total = []

        # Taget
        pos, orn = pb.getBasePositionAndOrientation(self.target)
        rpy = pb.getEulerFromQuaternion(orn)
        total.append(pos[0])
        total.append(pos[1])
        # total.append(rpy[-1])

        # Tool
        # pos, orn = pb.getBasePositionAndOrientation(self.tool)
        # rpy = pb.getEulerFromQuaternion(orn)
        # total.append(pos[0])
        # total.append(pos[1])
        # total.append(rpy[-1])

        obs = np.array(total)
        
        return obs

    ####################################################

    def render(self, action):


        # ------------------------------
        # Initial position
        x_start = 0.55 + 0.2 * action[0]
        y_start = 0.2 * action[1]
        yaw_start = action[2] * math.pi/2
        orn_start = quaternion_from_euler(0, 0, yaw_start)
        pb.resetBasePositionAndOrientation(self.tool, (x_start, y_start, 0.17), orn_start)


        # ------------------------------

        # Go approach
        info = pb.getBasePositionAndOrientation(self.tool)
        ee_pos1 = list(info[0])
        ee_pos1[-1] = ee_pos1[-1] + 0.14
        tool_orn = info[1]
        hook_rpy = euler_from_quaternion(tool_orn)
        ee_yaw1 = (hook_rpy[-1] - math.pi/4) % (np.sign(hook_rpy[-1] - math.pi/4)*(math.pi))
        ee_orn1 = quaternion_from_euler(math.pi, 0, ee_yaw1)
        approach_pose = (ee_pos1, ee_orn1)


        # Go grasp (down)
        info = pb.getBasePositionAndOrientation(self.tool)
        ee_pos2 = list(info[0])
        ee_pos2[-1] = ee_pos1[-1] - 0.04
        ee_orn2 = ee_orn1
        grasp_pose = (ee_pos2, ee_orn2)

        # Plan & execution
        end_pose_list = [approach_pose, grasp_pose]
        joint_states = pb.getJointStates(self.robot, list(range(pandaNumDofs)))# Given by Holding
        start_conf = []
        for i in joint_states:
            start_conf.append(i[0])
        success, trajectory = self.cartesian_planner.plan_multipose(start_conf, end_pose_list)
        for waypoint in trajectory:
            time.sleep(0.01) # Moving speed
            self.control.go_joint_space(self.robot, waypoint)
        self.control.finger_close(self.robot)
        # ------------------------------


        # Slide
        x_end = 0.55 + 0.2 * action[3] 
        y_end = 0.25 * action[4] 
        yaw_end = action[5] * math.pi *0.75 
        yaw_3 = yaw_end - math.pi/4  

        ee_orn3 = quaternion_from_euler(math.pi, 0, yaw_3)
        ee_pos3 = np.array([x_end, y_end, ee_pos2[-1]+0.03]).astype(float)
        goal_pose = (ee_pos3, ee_orn3)

        # Plan & execution
        end_pose_list = [goal_pose]
        joint_states = pb.getJointStates(self.robot, list(range(pandaNumDofs)))# Given by Holding
        start_conf = []
        for i in joint_states:
            start_conf.append(i[0])
        success, trajectory = self.cartesian_planner.plan_multipose(start_conf, end_pose_list)
        for waypoint in trajectory:
            time.sleep(0.01) # Moving speed
            finger_state = pb.getLinkState(self.robot, 11)[4]
            # if finger_state[-1] <= 0.05:
            self.control.finger_close(self.robot)
            self.control.go_joint_space(self.robot, waypoint)
        self.control.finger_open(self.robot)
        

        
        # ------------------------------
        # Up
        if success:
            ee_pos4 = ee_pos3
            ee_pos4[-1] = ee_pos4[-1] + 0.1
            ee_orn4 = ee_orn3
            up_pose = (ee_pos4, ee_orn4)
            info = pb.getBasePositionAndOrientation(self.tool)
            end_pose_list = [up_pose]
            joint_states = pb.getJointStates(self.robot, list(range(pandaNumDofs)))# Given by Holding
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
        joint_states = pb.getJointStates(self.robot, list(range(pandaNumDofs)))# Given by Holding
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
        
        hook_pose = pb.getBasePositionAndOrientation(self.tool)
        object_pose = pb.getBasePositionAndOrientation(self.target)


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
        # pb.resetBasePositionAndOrientation(self.scn.goal_mark, position, orn)

    
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

    pb.setRealTimeSimulation(1)
    wait_for_user()

    env.reset()
    wait_for_user()
    
    
    action = np.array((0,0,0,0,-1,0))
    env.render(action)

    wait_for_user()
    
    action = np.array((0,0,0,0,1,0))
    env.render(action)
    wait_for_user()

    action = np.array((0,-1,0,0,0,0))
    env.render(action)
    wait_for_user()

    action = np.array((0,1,0,0,0,0))
    env.render(action)
    wait_for_user()





