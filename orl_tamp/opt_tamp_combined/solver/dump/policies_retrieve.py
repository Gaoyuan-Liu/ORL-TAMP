
import os, sys, time
import math
import pybullet as p
import numpy as np
from gym import spaces
import rospy

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../')

# from agent import Agent
from opt_tamp_retrieve.env import RetrieveEnv
from opt_tamp_retrieve.motion_planner import CartesianPlanner


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion
from examples.pybullet.utils.pybullet_tools.franka_primitives import Trajectory, StateSubscriber, CmdPublisher, open_gripper, close_gripper, Reset, Observe


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, DDPG, SAC

file_path = os.path.dirname(os.path.realpath(__file__))

# sys.path.insert(0, file_path + '/../../')
# from camera.camera import Camera


#######################################################################
# Simulation
#######################################################################

class RetrieveEnvNominal(RetrieveEnv):
    def __init__(self, scn):
        super(RetrieveEnvNominal, self).__init__()
        self.scn = scn
        self.robot = scn.panda
        self.hook = scn.hook
        self.mode = 'eval'

        
    def reset(self, target, goal):
        self.target = target
        self.goal = goal
        self.objects = [self.target, self.hook]
        state = self.observe()
        self.episode_length = 0
        self.pre_dis = 0 #euclidean_distance(observation[:2], self.goal[:2])
        return state
    
    def reward(self, obs):
        dis = np.linalg.norm(obs[:2] - np.array([0,0]))
        if dis < 0.65:
            return 20, True
        else:
            return -1, False
    
    def connect(self):
        pass
    
  
    
#######################################################################

class RetrievePolicy():
    def __init__(self, scn):
        # self.initial_state = initial_state
        self.type = 'hook'
        self.scn = scn
        self.env = RetrieveEnvNominal(scn)

        # Load model
        # env = make_vec_env("Hooking-v0", n_envs=1)
        file = "./policies/retrieve/sac_retrieve"
        self.model = SAC.load(file, env=self.env)
       
    def specify(self, b, lg):
        self.target = b
        self.goal = np.array([lg.value[0][0], lg.value[0][1], 0])    

    def apply(self, _):
        p.setRealTimeSimulation(1)
        state = self.env.reset(self.target, self.goal)
        while True:
            action, _states = self.model.predict(state, deterministic=True)
            state, rewards, dones, info = self.env.step(action)

            if dones == True:
                break

        p.setRealTimeSimulation(0)





#######################################################################
# Real-world
#######################################################################

class RetrieveEnvNominalRealWorld(RetrieveEnv):
    def __init__(self):
        super(RetrieveEnvNominalRealWorld, self).__init__()
        # self.scn = scn
        # self.robot = scn.panda
        # self.hook = scn.hook
        self.mode = 'eval'
        # self.camera = Camera()
        
        # self.cup_id = '0'
        self.hook_id = '1'
        self.observe_prim = Observe()
        self.state_subscriber = StateSubscriber()
        self.cmd_publisher = CmdPublisher()
        self.motion_planner = CartesianPlanner()

        self.sim2real = {}

        
    def reset(self, target, goal):
        self.target = target
        self.observation = self.observe()
        self.episode_length = 0
        self.pre_dis = 0 
        return self.observation
    
    def observe(self):
        total = []
        pose_dict = self.observe_prim.apply([self.target, self.hook_id]) 
        cup_pose = pose_dict[self.target]
        hook_pose = pose_dict[self.hook_id]

        cup_euler = euler_from_quaternion(cup_pose[1])
        hook_euler = euler_from_quaternion(hook_pose[1])

        total.append(cup_pose[0][0])
        total.append(cup_pose[0][1])
        total.append(hook_pose[0][0])
        total.append(hook_pose[0][1])
        total.append(hook_euler[2])
        obs = np.array(total)
        return obs
    
    def render(self, action):

        # Interprete action
        d_x = 0.5 + 0.2 * action[0] #d_r * math.cos(d_theta)
        d_y = 0.4 * action[1] #d_r * math.sin(d_theta)
        d_yaw = action[2] * math.pi/2
        
        # Approach
        hook_pose = self.observe_prim.apply([self.hook_id])[self.hook_id] 
        hook_pose = ((hook_pose[0][0], hook_pose[0][1], 0.19), hook_pose[1])

        ee_pos1 = list(hook_pose[0])
        ee_pos1[-1] = ee_pos1[-1] + 0.15
        hook_orn = hook_pose[1]
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

        # Grasp
        ee_pos2 = list(hook_pose[0])
        ee_pos2[-1] = ee_pos1[-1] - 0.06
        ee_orn2 = ee_orn1
        grasp_pose = (ee_pos2, ee_orn2)
        joint_states = self.get_joint_states()
        success, path = self.cartesian_planner.plan_multipose(joint_states, [approach_pose, grasp_pose])
        if success:
            self.execute(path)
        else:
            print('No solution')
            return
        
        # Close
        close_gripper()


        # Slide
        x = d_x
        y = d_y
        yaw_3 = d_yaw #ee_yaw1 + d_yaw 
        ee_orn3 = quaternion_from_euler(math.pi, 0, yaw_3)
        ee_pos3 = np.array([x, y, ee_pos2[-1]+0.01]).astype(float)
        slide_pose = (ee_pos3, ee_orn3)
        joint_states = self.get_joint_states()
        success, path = self.cartesian_planner.plan_multipose(joint_states, [slide_pose])
        if success:
            self.execute(path)
        else:
            print('No solution')
            return
        
        # Open
        open_gripper()

        # Up
        ee_pos4 = ee_pos3
        ee_pos4[-1] = ee_pos4[-1] + 0.1
        ee_orn4 = ee_orn3
        up_pose = (ee_pos4, ee_orn4)
        joint_states = self.get_joint_states()
        success, path = self.cartesian_planner.plan_multipose(joint_states, [up_pose])
        if success:
            self.execute(path)
        else:
            print('No solution')
            return

        # Reset
        joint_states = self.get_joint_states()
        home_config = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
        path = self.cartesian_planner.plan_joint(joint_states, home_config)
        self.execute(path)


    def reward(self, observation):
        dis = np.linalg.norm(observation[:2] - np.array([0,0]))
        if dis < 0.7:
            return 20, True
        else:
            return -1, False

    # Utils
    def go(self, pose): # Pose has -math.pi/4 bias
        # Get joint state
        while not rospy.is_shutdown():
            joint_states = np.array(self.state_subscriber.joint_states.position)
            if len(joint_states) == 9:
                break
        path = self.motion_planner.plan(start_conf=joint_states, end_pose=pose)
        t = Trajectory(path)
        t.apply()

    def get_joint_states(self):
        while not rospy.is_shutdown():
            joint_states = np.array(self.state_subscriber.joint_states.position)
            if len(joint_states) == 9:
                break
        return joint_states
    
    def execute(self, path):
        t = Trajectory(path)
        t.apply()


    def connect(self):
        pass


########################################################

class RetrievePolicyRealWorld():
    def __init__(self):
        # self.initial_state = initial_state
        self.type = 'hook'
        # self.scn = scn
        self.env = RetrieveEnvNominalRealWorld()

        # Load model
        # env = make_vec_env("Hooking-v0", n_envs=1)
        file = "./policies/retrieve/sac_retrieve"
        self.model = SAC.load(file, env=self.env)
        self.reset = Reset()
       
    def specify(self, b, lg):
        self.target = '3'#'0' #self.env.sim2real[b]
        self.goal = np.array([lg.value[0][0], lg.value[0][1], 0])    

    def apply(self):
        state = self.env.reset(self.target, self.goal)
        while True:
            action, _states = self.model.predict(state, deterministic=True)
            state, rewards, dones, info = self.env.step(action)
            if dones == True:
                self.reset.apply()
                self.env.render(np.array([0.5, -0.5, 0]))
                # return state
                break

    

if __name__ == "__main__":
    # from common.scn import Scenario
    # import pybullet as pb
    # pb.connect(pb.GUI)
    # scn = Scenario()
    # env = RetrieveEnvNominalRealWorld(scn)

    # # env.go(pose=((0.4, 0.0, 0.65), quaternion_from_euler(math.pi, 0, -math.pi/4)))
    # env.render([0.7, 0.7, 0.7])
    rospy.init_node('hooking_policy', anonymous=True)
    # reset = Reset()
    # reset.apply()
