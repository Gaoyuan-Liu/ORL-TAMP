
import os, sys, time
import math
import pybullet as pb
import numpy as np
from gym import spaces
import rospy
import matplotlib.pyplot as plt


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../')

# from ..env import EdgePushEnv
# from ..pushing import Pushing
from opt_tamp_edgepush.env import EdgePushEnv
from opt_tamp_edgepush.pushing import Pushing


sys.path.insert(0, file_path + '/../../pddlstream/')
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion
from examples.pybullet.utils.pybullet_tools.franka_primitives import Trajectory, StateSubscriber, CmdPublisher, open_gripper, close_gripper, Reset, Observe, ObserveObject
from examples.pybullet.utils.pybullet_tools.utils import connect, create_obj, multiply, invert
from examples.pybullet.utils.pybullet_tools.panda_utils import get_edge_grasps

from stable_baselines3 import PPO, DDPG, SAC

sys.path.insert(0, file_path + '/../../')
from camera.camera import Camera
from common.scn import Scenario, load_ee


  
#######################################################################
# Simulation
#######################################################################

class EdgePushEnvNominal(EdgePushEnv):
    def __init__(self, scn):
        super(EdgePushEnvNominal, self).__init__()

        self.scn = scn
        self.robot = scn.robot
        self.target = scn.plate
        self.obstacles = [scn.low_table, scn.plate]
        self.body_type = 'cylinder'
        
   

    def reset(self):
        pb.changeDynamics(self.target, -1, lateralFriction=4)
        observation = self.observe()
        self.episodic_return = 0
        self.episode_length = 0
        return observation

    def connect(self):
        self.ee = load_ee(pos=(1,-1,1), mode='open')
        self.pusher = Pushing()

    
#######################################################################


class EdgePushPolicy():
    def __init__(self, scn):
        # self.initial_state = initial_state
        self.type = 'hook'
        self.scn = scn
        self.env = EdgePushEnvNominal(scn)
        file_path = os.path.dirname(os.path.realpath(__file__))
        file = file_path + "/policies/sac_edgepush"
        self.model = SAC.load(file, env=self.env)
       
    
    def specify(self, b, lg):
        self.env.target = b
        self.env.goal = np.array([lg.value[0][0], lg.value[0][1], 0])   
        
    def control(self):
        pb.setRealTimeSimulation(1)
        state = self.env.reset()
        while True:
            action, _states = self.model.predict(state, deterministic=True)
            state, rewards, dones, info = self.env.step(action)

            if dones == True:
                break

        pb.setRealTimeSimulation(0)



#######################################################################
# Real-world
#######################################################################

class EdgePushEnvNominalRealWorld(EdgePushEnv):
    def __init__(self, s2r):
        super(EdgePushEnvNominalRealWorld, self).__init__()

        
        self.observe_numerical = Observe()
        self.observe_object = ObserveObject()
        
        self.state_subscriber = StateSubscriber()
        self.cmd_publisher = CmdPublisher()

        self.s2r = s2r

        self.pose_dict = {}
        self.file_name = ''


    def reset(self):
        self.target_aruco = self.s2r[self.target]
        state = self.observe()
        self.episodic_return = 0
        self.episode_length = 0
        close_gripper()
        return state
    
    def render(self, action):
        # return
        plate_pose = self.pose_dict[self.target_aruco]
        x_0 = plate_pose[0][0] 
        y_0 = plate_pose[0][1]

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


        theta = theta_1 + theta_2 + (2 * math.pi-theta_2_range)/2
        theta = theta % (2 * math.pi)

        x_3 = x_1 + l * math.cos(theta)
        y_3 = y_1 + l * math.sin(theta)

        start_point = np.array([x_1, y_1])
        end_point = np.array([x_3, y_3])

        ##############################################

            
        initial_conf = self.get_joint_states()
        points = self.pusher.push_plan(start_point, end_point, [], initial_conf)

        if points is None:
            return
        
        path = []
        for point in points:
            path.append(point.positions)
        self.execute(path)
        

    def observe(self):

        self.pose_dict = self.observe_numerical.control([self.target_aruco]) 
        plate_pose = self.pose_dict[self.target_aruco]
        depth_img = self.observe_object.control(plate_pose)
        plt.imshow(depth_img)
        # plt.show()
        plt.savefig('depth_img.png')


        return depth_img #observation
    

    def reward(self):
        reward = -1
        done = False

        # self.pose_dict = self.observe_numerical.control([self.target_aruco]) 
        plate_pose = self.pose_dict[self.target_aruco]

        # ------------------------------------------------------
        # file_path = os.path.dirname(os.path.realpath(__file__))
        # print(f'file_path: {file_path}')
        # try:
        #     os.remove('./hull_background.obj')
        #     os.remove('./hull.obj')
        # except FileNotFoundError:
        #     print('File not found')

        
        self.file_name = str(self.episode_length) + 'hull'
        self.observe_object.reconstruct(plate_pose, file_name=self.file_name)

        # input('Press enter to continue: ')

        plate = create_obj('./' + self.file_name + '.obj')
        obstacle = create_obj('./' + self.file_name + '_background.obj')
        

        # input('Press enter to continue: ')

        pb.resetBasePositionAndOrientation(obstacle, [1,1,0.5], [0,0,0,1])
        pb.resetBasePositionAndOrientation(plate, [1,1,0.5], [0,0,0,1])

        # ------------------------------------------------------
        # Can be grasped
        ee = load_ee(pos=(1,-1,1), mode='open')
        obstacles = [obstacle]
        grasps = get_edge_grasps(plate)
        for grasp in grasps:
            grasp_pose = multiply(((1,1,0.5), (0,0,0,1)), invert(grasp)) 

            pb.setRealTimeSimulation(0)
            pb.resetBasePositionAndOrientation(ee, grasp_pose[0], grasp_pose[1])
            pb.performCollisionDetection()

            # input('Press enter to continue: ')
            
            collision = False
            for i in obstacles:
                contact_points = pb.getContactPoints(ee, i)
                if len(contact_points) > 0:
                    collision = True
                    break
            if collision == False: 
                print(' Goal Reached!')
                reward = 10
                done = True
                return reward, done
                # break

        pb.removeBody(obstacle)
        pb.removeBody(plate)
        pb.removeBody(ee)   
        
        try:
            os.remove('./' + self.file_name + '.obj')
            os.remove('./' + self.file_name + '_background.obj')
        except FileNotFoundError:
            print('File not found')

        return reward, done 


    def execute(self, path):
        t = Trajectory(path)
        t.control()

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
    def __init__(self, s2r):

        self.type = 'hook'
        self.env = EdgePushEnvNominalRealWorld(s2r)

        file_path = os.path.dirname(os.path.realpath(__file__))
        file = file_path + "/policies/sac_edgepush"
        self.model = SAC.load(file, env=self.env)
        # self.reset = Reset()
        
    def specify(self, target, lg):
        self.env.target = target
        self.env.goal = [lg.value[0][0], lg.value[0][1], 0]   

    def control(self):
        state = self.env.reset()
        while True:
            action, _states = self.model.predict(state, deterministic=True)
            state, rewards, dones, info = self.env.step(action)

            if dones == True:
                os.rename("./" + self.env.file_name + ".obj", './hull_final.obj')
                os.rename("./" + self.env.file_name + "_background.obj", './hull_background_final.obj')
                break


if __name__ == "__main__":
    rospy.init_node('edgepushing_policy', anonymous=True)
    pb.connect(pb.GUI)

    env = EdgePushEnvNominalRealWorld()

    env.render([1,0.5])