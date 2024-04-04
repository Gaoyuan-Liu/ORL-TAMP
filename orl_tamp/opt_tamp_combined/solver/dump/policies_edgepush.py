
import os, sys, time
import math
import pybullet as pb
import numpy as np
from gym import spaces
import rospy
import matplotlib.pyplot as plt


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../')

from opt_tamp_push.env import EdgePushingEnv
from opt_tamp_push.pushing import Pushing
from utils_h2rl import euclidean_distance



file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion
from examples.pybullet.utils.pybullet_tools.franka_primitives import Trajectory, StateSubscriber, CmdPublisher, open_gripper, close_gripper, Reset, Observe, ObserveObject
from examples.pybullet.utils.pybullet_tools.utils import connect, create_obj, multiply, invert
from examples.pybullet.utils.pybullet_tools.panda_utils import get_edge_grasps

from stable_baselines3 import PPO, DDPG, SAC

sys.path.insert(0, file_path + '/../../')
from camera.camera import Camera
from common.scn import Scenario, load_ee


class EdgePushingEnvNominal(EdgePushingEnv):
    def __init__(self, scn):
        super(EdgePushingEnvNominal, self).__init__()

        self.scn = scn
        self.robot = scn.panda
        self.table = scn.low_table
        self.dish = scn.dish
        
   

    def reset(self, target, goal):
        pb.changeDynamics(self.dish, -1, lateralFriction=1)
        self.goal = goal
        self.target = target
        self.objects = [self.target]
        state = self.observe()
        self.episodic_return = 0
        self.episode_length = 0
        return state
    

    def connect(self):
        self.body_type = 0
        self.pusher = Pushing()
        self.ee = load_ee(pos=(1,-1,1), mode='open')



    

  
#######################################################################
# Simulation
#######################################################################

class EdgePushPolicy():
    def __init__(self, scn):
        # self.initial_state = initial_state
        self.type = 'push'
        # self.scn = scn
        self.env = EdgePushingEnvNominal(scn)

        file = "./policies/edgepush/sac_edgepush"
        self.model = SAC.load(file, env=self.env)
       
    
    def specify(self, b, lg):
        self.target = b
        self.goal = lg.value #np.array([lg.value[0][0], lg.value[0][1], 0])   
        
    def apply(self, _):
        pb.setRealTimeSimulation(1)
        state = self.env.reset(self.target, self.goal)
        while True:
            action, _states = self.model.predict(state, deterministic=True)
            state, rewards, dones, info = self.env.step(action)

            if dones == True:
                pb.resetBasePositionAndOrientation(self.env.target, (0.4, 0.23, 0.17), (0,0,0,1))
                break

        pb.setRealTimeSimulation(0)



#######################################################################
# Real-world
#######################################################################

class EdgePushingEnvNominalRealWorld(EdgePushingEnv):
    def __init__(self):
        super(EdgePushingEnvNominalRealWorld, self).__init__()

        self.sim2real = {4:'3'}
        self.focus_sim = 4
        self.focus = self.sim2real[self.focus_sim]
        self.observe_numerical = Observe()
        self.observe_object = ObserveObject()
        

        self.state_subscriber = StateSubscriber()
        self.cmd_publisher = CmdPublisher()


    def reset(self, goal):
        self.goal = goal
        state = self.observe()
        # self.pre_dis = euclidean_distance(state[0:2], self.goal[0:2])
        self.episodic_return = 0
        self.episode_length = 0
        return state
    
    def render(self, action):
        pose_dict = self.observe_numerical.apply([self.focus]) 
        dish_pose = pose_dict[self.focus]
        x_0 = dish_pose[0][0] 
        y_0 = dish_pose[0][1]

        r_start = 0.15
        l_max = 0.2

        theta = action[0] * 2 * math.pi
        l = action[1] * l_max

        x_1 = r_start * math.cos(theta) + x_0
        y_1 = r_start * math.sin(theta) + y_0

        x_2 = x_1 - l * math.cos(theta)
        y_2 = y_1 - l * math.sin(theta)

        start_point = np.array([x_1, y_1])
        end_point = np.array([x_2, y_2])

        close_gripper()    
        initial_conf = self.get_joint_states()
        points = self.pusher.push_plan(start_point, end_point, [], initial_conf)

        if points is None:
            return
        
        path = []
        for point in points:
            path.append(point.positions)
        self.execute(path)
        open_gripper()

    def observe(self):
        
        pose_dict = self.observe_numerical.apply([self.focus]) 
        dish_pose = pose_dict[self.focus]
        depth_img = self.observe_object.apply(dish_pose)
        plt.imshow(depth_img)
        # plt.show()
        plt.savefig('depth_img.png')


        return depth_img #observation
    

    def reward(self):
        reward = -1
        done = False

        pose_dict = self.observe_numerical.apply([self.focus]) 
        dish_pose = pose_dict[self.focus]


        # ------------------------------------------------------
        self.observe_object.reconstruct(dish_pose)
        file_path = os.path.dirname(os.path.realpath(__file__)) + '/../../camera/'
        # sys.path.insert(0, file_path + '/../../camera/')
        obstacle = create_obj('./hull_background.obj')
        dish = create_obj('./hull.obj')
        # pb.resetBasePositionAndOrientation(obstacle, np.array(dish_pose[0]), (0,0,0,1))
        # pb.resetBasePositionAndOrientation(self.focus_sim, np.array(dish_pose[0])+ np.array([0,0,0.01]), (0,0,0,1))

        pb.resetBasePositionAndOrientation(obstacle, [1,1,0.5], [0,0,0,1])
        pb.resetBasePositionAndOrientation(dish, [1,1,0.5], [0,0,0,1])

        input('Press enter to continue: ')

        # ------------------------------------------------------
        # Can be grasped
        ee = load_ee(pos=(1,-1,1), mode='open')
        obstacles = [obstacle]
        grasps = get_edge_grasps(dish)
        for grasp in grasps:
            grasp_pose = multiply(dish_pose, invert(grasp)) 

            pb.setRealTimeSimulation(0)
            pb.resetBasePositionAndOrientation(ee, grasp_pose[0], grasp_pose[1])
            pb.performCollisionDetection()
            
            collision = False
            for i in obstacles:
                contact_points = pb.getContactPoints(ee, i)
                if len(contact_points) > 0:
                    collision = True
                    break
            if collision == False: 
                print(' Goal Reached!')
                done = True
                break

        pb.removeBody(obstacle)
        pb.removeBody(dish)
        pb.removeBody(ee)    

        return reward, done 


    def execute(self, path):
        t = Trajectory(path)
        t.apply()

    def connect(self):
        self.pusher = Pushing()

    def get_joint_states(self):
        while not rospy.is_shutdown():
            joint_states = np.array(self.state_subscriber.joint_states.position)
            if len(joint_states) == 9:
                break
        return joint_states


#######################################################################

class EdgePushPolicyRealWorld():
    def __init__(self):

        self.type = 'edgepush'
        self.env = EdgePushingEnvNominalRealWorld()

        file = "./policies/edgepush/sac_edgepush"
        self.model = SAC.load(file, env=self.env)
        # self.reset = Reset()
        
    def specify(self, focus_sim, lg):
        self.env.focus_sim = focus_sim
        self.goal = lg.value #[lg.value[0][0], lg.value[0][1], 0]   

    def apply(self):
        state = self.env.reset(self.goal)
        while True:
            action, _states = self.model.predict(state, deterministic=True)
            state, rewards, dones, info = self.env.step(action)

            

            if dones == True:
                break


if __name__ == "__main__":
    rospy.init_node('edgepushing_policy', anonymous=True)
    pb.connect(pb.GUI)

    env = EdgePushingEnvNominalRealWorld()

    env.render([1,0.5])