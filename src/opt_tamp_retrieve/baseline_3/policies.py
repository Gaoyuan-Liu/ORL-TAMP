
import os, sys, time
import math
import pybullet as p
import numpy as np
from gym import spaces

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')

from agent import Agent
from env import HookingEnv
from utils_h2rl import euclidean_distance
from cartesian_planner import CartesianPlanner
from control import Control


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion

from opt_solver import Solver
from plan_executor import execute


class HookEnvNominal():
    def __init__(self, scn):

        self.scn = scn
        self.hook = scn.hook
        self.robot = scn.panda
        self.pandaNumDofs = 7
        self.n_cube = 1
        self.n_tool = 1
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float64), high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float64))
        self.observation_space = spaces.Box(low=np.array([[-1.2, -1.2, -math.pi]]*(self.n_cube+self.n_tool), dtype=np.float64), high=np.array([[1.2, 1.2, math.pi]]*(self.n_cube+self.n_tool), dtype=np.float64))
        self.goal = np.array([0,0,0])
        self.cartesian_planner = CartesianPlanner()
        self.control = Control(self.robot)
        self.solver = Solver(robot = scn.panda, 
                             movable=[scn.hook], 
                             tools=[scn.hook], 
                             surfaces=[scn.goal_surface, scn.hook_mark], 
                             marks=[scn.hook_mark])
        

    def reset(self, focused_object, goal):
        self.focused_object = focused_object
        self.goal = goal
        self.objects = [self.focused_object, self.hook]

    def step(self, action):
        self.render(action)
        observation = self.observe()
        done = self.done(observation)
        reward = 0

        return observation, reward, done, None

    def render(self, action):

        # Initial position
        
        theta_0 = action[0] * math.pi/2
        r_0 = 0.55 + action[1] * 0.15
        yaw_0 = action[2] * math.pi/2
        x_0 = r_0 * math.cos(theta_0)
        y_0 = r_0 * math.sin(theta_0)
        pos_0 = [x_0, y_0, 0.02]
        orn_0 = quaternion_from_euler(0,0,yaw_0)
        # p.resetBasePositionAndOrientation(self.objects[-1], pos_0, orn_0)

        p.resetBasePositionAndOrientation(self.scn.hook_mark, pos_0, orn_0)
        goal_0 = ('on', self.scn.hook, self.scn.hook_mark)

        p.setRealTimeSimulation(0)
        plan, cost, evaluations = self.solver.solve(goal_0, collisions=False)

        execute(self.solver.problem, plan, self.scn)
        p.setRealTimeSimulation(1)



        ########################################
        d_theta = action[3] * math.pi/2
        d_r = 0.55 + action[4] * 0.15
        d_yaw = action[5] * math.pi/2

        d_x = d_r * math.cos(d_theta)
        d_y = d_r * math.sin(d_theta)

        # Go approach
        info = p.getBasePositionAndOrientation(self.objects[-1])
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

        # Go grasp
        info = p.getBasePositionAndOrientation(self.objects[-1])
        ee_pos2 = list(info[0])
        ee_pos2[-1] = ee_pos1[-1] - 0.04
        ee_orn2 = ee_orn1
        grasp_pose = (ee_pos2, ee_orn2)

        # Move
        observation = self.observe()
       
        x = d_x
        y = d_y
        yaw_3 = ee_yaw1 + d_yaw 

        ee_orn3 = quaternion_from_euler(math.pi, 0, yaw_3)
        ee_pos3 = np.array([x, y, 0.15]).astype(float)
        goal_pose = (ee_pos3, ee_orn3)

        # Plan & execution
        end_pose_list = [approach_pose, grasp_pose, goal_pose]
        joint_states = p.getJointStates(self.robot, list(range(self.pandaNumDofs)))# Given by Holding
        start_conf = []
        for i in joint_states:
            start_conf.append(i[0])
        success, trajectory = self.cartesian_planner.plan_multipose(start_conf, end_pose_list)
        for waypoint in trajectory:
            time.sleep(0.01) # Moving speed
            finger_state = p.getLinkState(self.robot, 11)[4]
            if finger_state[-1] <= 0.05:
                self.control.finger_close()
            self.control.go_joint_space(waypoint)
        self.control.finger_open()
        

        
        # Up
        if success:
            ee_pos4 = ee_pos3
            ee_pos4[-1] = ee_pos4[-1] + 0.1
            ee_orn4 = ee_orn3
            up_pose = (ee_pos4, ee_orn4)
            info = p.getBasePositionAndOrientation(self.objects[-1])
            end_pose_list = [up_pose]
            joint_states = p.getJointStates(self.robot, list(range(self.pandaNumDofs)))# Given by Holding
            start_conf = []
            for i in joint_states:
                start_conf.append(i[0])
            _, trajectory = self.cartesian_planner.plan_multipose(start_conf, end_pose_list)
            for waypoint in trajectory:
                time.sleep(0.01) # Moving speed
                finger_state = p.getLinkState(self.robot, 11)[4]
                self.control.go_joint_space(waypoint)
                self.control.finger_open()

        # Reset
        joint_states = p.getJointStates(self.robot, list(range(self.pandaNumDofs)))# Given by Holding
        start_conf = []
        for i in joint_states:
            start_conf.append(i[0])
        default_config = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
        trajectory = self.cartesian_planner.plan_joint(start_conf, default_config)
        for waypoint in trajectory:
            time.sleep(0.02) # Moving speed
            self.control.go_joint_space(waypoint)
        # render the action
        pass

    def done(self, obs):
        done = False
        
        for i in range(self.n_cube):
            # dis = euclidean_distance(obs[i][:2], self.goal[:2])
            # if dis <= 0.15:
            #     done = True
            #     print('\n Success! ')

            dis = euclidean_distance(obs[i][:2], np.array([0,0]))
            if dis <= 0.6 and dis >= 0.3:
                done = True
                print('\n Success in nominal! ')
        return done
    
    def observe(self):
        total = []
        for object in self.objects:
            pose, o = p.getBasePositionAndOrientation(object)
            rpy = p.getEulerFromQuaternion(o)
            i_ = list(pose[:2])
            i_.append(rpy[-1])
            total.append(i_)
        return np.array(total, dtype=np.float64)
    
    def render_goal(self, goal):
        goal = goal.reshape((-1,))
        position = [goal[0], goal[1], 0.0005] 
        orn = [0,0,0,1]#quaternion_from_euler(0, 0, goal[2])
        # p.resetBasePositionAndOrientation(self.scn.goal_surface, position, orn)


    
#######################################################################

class HookPolicy():
    def __init__(self, scn):
        # self.initial_state = initial_state
        self.type = 'hook'
        self.scn = scn
        self.env = HookEnvNominal(scn)
        state_dim = self.env.observation_space.sample().size
        action_dim = self.env.action_space.shape[0]
        goal_dim = self.env.goal.size
        H = 5
        render = True
        lr = 0.001
        self.agent = Agent(self.env, H, state_dim, action_dim, goal_dim, render, lr)
        # Action space
        action_clip_low = np.array([-1] * action_dim)
        action_clip_high = np.array([1] * action_dim)
        exploration_action_noise = np.array([0.1] * action_dim)       
        H = 20                   
        lamda = 0.3               
        epsilon = 0.0 # exploration            
        # DDPG parameters:
        gamma = 0.97                
        lr = 0.001
        self.agent.set_parameters(epsilon, lamda, gamma, action_clip_low, action_clip_high, exploration_action_noise)

        # Trained models
        action_name = 'hook'
        directory = "./preTrained/{}".format(action_name) 
        filename = "policy_{}".format(action_name)
        self.agent.load(directory, filename)


    # def pre_statege(self):
    #     # put the object in the initial state: hook and focused object
    def specify(self, focused_object, goal_p):
        self.focused_object = focused_object
        self.goal = np.array([goal_p.value[0][0], goal_p.value[0][1], 0])    

    def apply(self, _):
        p.setRealTimeSimulation(1)
        self.env.reset(self.focused_object, self.goal)
        state = self.env.observe()
        state = state.reshape(-1)

        last_state = self.agent.run(state, self.env.goal)
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
