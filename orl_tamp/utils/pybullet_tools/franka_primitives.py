#!/usr/bin/env python

# Run this in the terminal:
# source /home/liu/catkin_ws/devel/setup.bash

import time, math, os, sys  

from itertools import count
import numpy as np
import pybullet as pb

from scipy.spatial.transform import Rotation

from .pr2_utils import get_top_grasps
from .utils import get_pose, set_pose, get_movable_joints, \
    set_joint_positions, add_fixed_constraint, enable_real_time, disable_real_time, joint_controller, \
    enable_gravity, get_refine_fn, wait_for_duration, link_from_name, get_body_name, multiply, multiply_quats

from .panda_primitives import motion_planner
from .transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Pose as Pose_msg 

# TODO: deprecate

# ROS
import rospy
from franka_gripper.msg import MoveAction, GraspEpsilon, MoveGoal, GraspAction, GraspGoal, GraspActionGoal
import actionlib
from std_msgs.msg import Float64, Int64, Float64MultiArray
from sensor_msgs.msg import JointState

from scipy.interpolate import interp1d


# Moveit
import moveit_commander
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import Pose

import roboticstoolbox as rtb
robot = rtb.models.Panda()


##################################################



class StateSubscriber():
  def __init__(self):
    JOINT_STATES_TOPIC = '/joint_states'
    self.joint_states = JointState()
    self.sub_joint_states = rospy.Subscriber(JOINT_STATES_TOPIC, JointState, self.sub_joint_states_callback)

  def sub_joint_states_callback(self, data):
    self.joint_states = data

class CmdPublisher():
  def __init__(self):
    self.joint1_pub = rospy.Publisher('/joint1_position_controller/command', Float64, queue_size=10)
    self.joint2_pub = rospy.Publisher('/joint2_position_controller/command', Float64, queue_size=10)
    self.joint3_pub = rospy.Publisher('/joint3_position_controller/command', Float64, queue_size=10)
    self.joint4_pub = rospy.Publisher('/joint4_position_controller/command', Float64, queue_size=10)
    self.joint5_pub = rospy.Publisher('/joint5_position_controller/command', Float64, queue_size=10)
    self.joint6_pub = rospy.Publisher('/joint6_position_controller/command', Float64, queue_size=10)
    self.joint7_pub = rospy.Publisher('/joint7_position_controller/command', Float64, queue_size=10)

    self.cmd_1 = Float64()
    self.cmd_2 = Float64()
    self.cmd_3 = Float64()
    self.cmd_4 = Float64()
    self.cmd_5 = Float64()
    self.cmd_6 = Float64()
    self.cmd_7 = Float64() 

  def publish(self, positions):
    self.cmd_1.data = positions[0]
    self.cmd_2.data = positions[1]
    self.cmd_3.data = positions[2]
    self.cmd_4.data = positions[3]
    self.cmd_5.data = positions[4]
    self.cmd_6.data = positions[5]
    self.cmd_7.data = positions[6]

    self.joint1_pub.publish(self.cmd_1)
    self.joint2_pub.publish(self.cmd_2)
    self.joint3_pub.publish(self.cmd_3)
    self.joint4_pub.publish(self.cmd_4)
    self.joint5_pub.publish(self.cmd_5)
    self.joint6_pub.publish(self.cmd_6)
    self.joint7_pub.publish(self.cmd_7)



##################################################
# For control in the real world
##################################################
 
def open_gripper(width=0.8):
    joint_states = [width/2, width/2]
    goal = MoveGoal(width=float(width), speed=float(0.1))
    release_client = actionlib.SimpleActionClient('franka_gripper/move', MoveAction)
    release_client.wait_for_server()
    release_client.send_goal(goal)
    release_client.wait_for_result(rospy.Duration.from_sec(5.0))  

def close_gripper():
    GRASP_EPSILON = GraspEpsilon(inner=0.039, outer=0.039)
    goal = GraspGoal(width=float(0.04), epsilon=GRASP_EPSILON, speed=float(0.1), force=float(100))
    grasp_client = actionlib.SimpleActionClient('franka_gripper/grasp', GraspAction)
    grasp_client.wait_for_server()
    grasp_client.send_goal(goal)
    grasp_client.wait_for_result(rospy.Duration.from_sec(5.0))



class GripperCommand():
    def __init__(self, robot, arm, mode, teleport=False): # Two possible modes: 'close' and 'open'
        self.robot = robot
        self.arm = arm
        self.mode = mode
        self.teleport = teleport
        self.type = 'GripperCommand'

    def open_gripper(self, width=0.8):
        joint_states = [width/2, width/2]
        goal = MoveGoal(width=float(width), speed=float(0.1))
        release_client = actionlib.SimpleActionClient('franka_gripper/move', MoveAction)
        release_client.wait_for_server()
        release_client.send_goal(goal)
        release_client.wait_for_result(rospy.Duration.from_sec(5.0))  

    def close_gripper(self):
        GRASP_EPSILON = GraspEpsilon(inner=0.039, outer=0.039)
        goal = GraspGoal(width=float(0.04), epsilon=GRASP_EPSILON, speed=float(0.1), force=float(100))
        grasp_client = actionlib.SimpleActionClient('franka_gripper/grasp', GraspAction)
        grasp_client.wait_for_server()
        grasp_client.send_goal(goal)
        grasp_client.wait_for_result(rospy.Duration.from_sec(5.0))

    def control(self):
        if self.mode == 'close':
            self.close_gripper()
        elif self.mode == 'open':
            self.open_gripper()
        else:
            raise ValueError(self.mode)
        
        
    # def control(self, **kwargs):
    #     joints = get_gripper_joints(self.robot, self.arm)
    #     positions = [self.position]*len(joints)
    #     for _ in joint_controller_hold(self.robot, joints, positions):
    #         yield

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, get_body_name(self.robot), self.mode) 
    


##################################################

class Attach():
    def __init__(self, robot, arm, grasp, body):
        self.robot = robot
        self.arm = arm
        self.grasp = grasp
        self.body = body
        self.type = 'Attach'

    def control(self):
        print('Attach.control()')

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_body_name(self.body))
    

##################################################

class Detach():
    def __init__(self, robot, arm, body):
        self.robot = robot
        self.arm = arm
        self.body = body
        self.link = link_from_name(self.robot, 'panda_hand') #link_from_name(self.robot, PANDA_TOOL_FRAMES.get(self.arm, self.arm))
        self.type = "Detach"
        # TODO: pose argument to maintain same object
    def control(self):
        print('Detach.control()')

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_body_name(self.body))

    
##################################################

class Trajectory():
    def __init__(self, path):
        self.cmd_publisher = CmdPublisher()
        self.state_subscriber = StateSubscriber()
        self.path = np.asarray(path)[:, :7]

        self.type = 'Trajectory'
        self.control_freq = 100
        self.dt = 1.0 / self.control_freq
        self.qdmax = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])/10

    def control(self, tacc=0.0001):
        # Make trajectory
        # time_vector = np.linspace(0, duration, len(self.path)) # (start_time, end_time(s), len(waypoints))
        # joint_trajectory = interp1d(time_vector, self.path, axis=0, kind='cubic')

        traj = rtb.mstraj(np.array(self.path), self.dt, tacc=tacc, qdmax=self.qdmax)

        # Loop
        rate = rospy.Rate(self.control_freq)

        start_time = rospy.get_time()

        time_step = 0
        while not rospy.is_shutdown() and time_step < len(traj.q):
            # current_time = rospy.get_time() - start_time
            current_waypoint = traj.q[time_step]
            
            self.cmd_publisher.publish(current_waypoint[:7])
            time_step += 1
            rate.sleep()

    def calculate_duration(self, path, max_velocity=1):
        accumulated_distance = np.zeros(len(path[0]))
        for i in range(len(path)):
            for j in range(7):
                if i == 0:
                    accumulated_distance[j] += (np.linalg.norm(np.array(path[i]), ord=np.inf))
                else:
                    accumulated_distance[j] += (np.linalg.norm(np.array(path[i]) - np.array(path[i-1]), ord=np.inf))
            
        max_distance = np.max(accumulated_distance)
        duration = max_distance / max_velocity
        return duration


    def reverse(self):
        reversed_path = []
        for i in reversed(self.path):
            reversed_path.append(i)
        return Trajectory(reversed_path)

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, 'franka', len(self.path))

##################################################
# Reset

class Reset():
    def __init__(self) -> None:
        self.home_config = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
        self.obstacles = []
        self.state_subscriber = StateSubscriber()
        self.cmd_publisher = CmdPublisher()
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.move_group.set_planner_id('RRTConnectkConfigDefault')

    def paln(self):
        joint_states = self.get_joint_states()
        path = motion_planner(start_config=joint_states, end_config=self.home_config)
        return path
    
    def paln_moveit(self):
        # joint_states = self.get_joint_states()
        # path = motion_planner(start_config=joint_states, end_pose=ee_pose)	
        
        start_conf = self.get_joint_states()

        # if np.linalg.norm(start_conf[:7] - self.home_config) < 0.1:
        #     return None
        robot_state = RobotState()
        robot_state.joint_state.name = self.move_group.get_joints()[:7]
        robot_state.joint_state.position = start_conf[:7]
        self.move_group.set_start_state(robot_state)


        self.move_group.set_joint_value_target(self.home_config)
        ourput = self.move_group.plan()
        success = ourput[0]
        print('success = ', success)
        robot_trajectory = ourput[1]

        # input('Press enter to continue: ')

        # Save trajectory
        path = []
        if success:
            for point in robot_trajectory.joint_trajectory.points:
                path.append(point.positions)
        else:
            success = False

        return path
    
    def control(self):
        
        path = self.paln_moveit()
        t = Trajectory(path)
        t.control()
        open_gripper()

    def get_joint_states(self):
        while not rospy.is_shutdown():
            joint_states = np.array(self.state_subscriber.joint_states.position)
            if len(joint_states) == 9:
                break
        return joint_states

##################################################
# Obseve
class Observe_Numerical():
    def __init__(self) -> None:
        
        self.obstacles = []
        self.state_subscriber = StateSubscriber()
        self.cmd_publisher = CmdPublisher()

        file_path = os.path.dirname(os.path.realpath(__file__))
        sys.path.insert(0, file_path + '/../../../../../') #src
        from camera.camera import Camera
        self.camera = Camera()

        self.ee_pose_list_moveit = [((0.4, 0.0, 0.8), quaternion_from_euler(math.pi, 0, -math.pi/4)),
                             ((0.5, 0.0, 0.7), quaternion_from_euler(math.pi, -math.pi/4, -math.pi/6, 'sxzy')),
                             ((0.3, 0.0, 0.75), quaternion_from_euler(math.pi, -math.pi/4, -math.pi/4, 'sxzy')),
                             ((0.5, 0.0, 0.7), quaternion_from_euler(math.pi/4, math.pi-math.pi/6, 0, 'szxy')),
                             ((0.5, 0.0, 0.7), quaternion_from_euler(math.pi/4, math.pi+math.pi/6, 0, 'szxy'))]
        
        self.ee_pose_list = [((0.4, 0.0, 0.8), quaternion_from_euler(math.pi, 0, 0)),
                             ((0.5, 0.0, 0.7), quaternion_from_euler(math.pi, -math.pi/6, 0)),
                             ((0.3, 0.0, 0.75), quaternion_from_euler(math.pi, -math.pi/4, 0)),
                             ((0.5, 0.0, 0.7), quaternion_from_euler(0, math.pi-math.pi/6, 0, 'szxy')),
                             ((0.5, 0.0, 0.7), quaternion_from_euler(0, math.pi+math.pi/6, 0, 'szxy'))]
        
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.move_group.set_planner_id('RRTConnectkConfigDefault')

    def paln(self, pose, eef_step=0.001):
        # joint_states = self.get_joint_states()
        # path = motion_planner(start_config=joint_states, end_pose=ee_pose)

    
        start_conf = self.get_joint_states()
        robot_state = RobotState()
        robot_state.joint_state.name = self.move_group.get_joints()[:7]
        robot_state.joint_state.position = start_conf[:7]
        self.move_group.set_start_state(robot_state)

        # Formulize end pose
        waypoints = []

        end_pose_msg = Pose_msg()
        end_pose_msg.position.x = pose[0][0]
        end_pose_msg.position.y = pose[0][1]
        end_pose_msg.position.z = pose[0][2]
        end_pose_msg.orientation.x = pose[1][0]
        end_pose_msg.orientation.y = pose[1][1]
        end_pose_msg.orientation.z = pose[1][2]
        end_pose_msg.orientation.w = pose[1][3]

            # Set Cartesian path
        waypoints.append(end_pose_msg)

        (plan_waypoints, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step, jump_threshold=5.0)


        # Save trajectory
        path = []
        if fraction > 0.7:
            success = True
            for point in plan_waypoints.joint_trajectory.points:
                path.append(point.positions)
        else:
            success = False

        return path



    def control(self, expected_ids=[]):
        pose_dict = {}
        while True:

            for i in range(len(self.ee_pose_list_moveit)):
                path = self.paln(self.ee_pose_list_moveit[i])
                t = Trajectory(path)
                t.control()
                pose_dict_temporary = self.camera.aruco_pose_detection(ee_pose=self.ee_pose_list[i])
                if len(set(list(pose_dict_temporary.keys())) - set(list(pose_dict.keys()))) != 0:
                    pose_dict.update(pose_dict_temporary)

                print(' detected_ids = ', list(pose_dict.keys()))
                print(' expected_ids = ', expected_ids)     

                if len(set(expected_ids) - set(list(pose_dict.keys()))) == 0:
                    break

            if len(set(expected_ids) - set(list(pose_dict.keys()))) == 0:
                break
        return pose_dict
    
    def get_joint_states(self):
        while not rospy.is_shutdown():
            
            joint_states = np.array(self.state_subscriber.joint_states.position)
            if len(joint_states) == 9:
                break
            print(' Waiting for joint states...')
            time.sleep(0.1)
        return joint_states
    
    
class Observe(Observe_Numerical):
    pass

##################################################
# 

class ObserveObject():
    def __init__(self) -> None:
        
        self.obstacles = []
        self.state_subscriber = StateSubscriber()
        self.cmd_publisher = CmdPublisher()

        file_path = os.path.dirname(os.path.realpath(__file__))
        sys.path.insert(0, file_path + '/../../../../../') #src
        from camera.camera import Camera
        self.camera = Camera()
        
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.move_group.set_planner_id('RRTConnectkConfigDefault')

    def paln(self, pose, eef_step=0.001):
        # joint_states = self.get_joint_states()
        # path = motion_planner(start_config=joint_states, end_pose=ee_pose)

    
        start_conf = self.get_joint_states()
        robot_state = RobotState()
        robot_state.joint_state.name = self.move_group.get_joints()[:7]
        robot_state.joint_state.position = start_conf[:7]
        self.move_group.set_start_state(robot_state)

        # Formulize end pose
        waypoints = []

        end_pose_msg = Pose_msg()
        end_pose_msg.position.x = pose[0][0]
        end_pose_msg.position.y = pose[0][1]
        end_pose_msg.position.z = pose[0][2]
        end_pose_msg.orientation.x = pose[1][0]
        end_pose_msg.orientation.y = pose[1][1]
        end_pose_msg.orientation.z = pose[1][2]
        end_pose_msg.orientation.w = pose[1][3]

            # Set Cartesian path
        waypoints.append(end_pose_msg)

        (plan_waypoints, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step, jump_threshold=5.0)

        # Save trajectory
        path = []
        if fraction > 0.7:
            success = True
            for point in plan_waypoints.joint_trajectory.points:
                path.append(point.positions)
        else:
            success = False
        return path

    def control(self, object_pose, observe_height=0.35):
        # TODO: Need to add bias here
        bias = np.array([-0.05, -0.03, 0.1 + 0.08])
        path = self.paln(((object_pose[0][0] + bias[0], object_pose[0][1] + bias[1], observe_height + bias[2]), quaternion_from_euler(math.pi, 0, -math.pi/4)))
        t = Trajectory(path)
        t.control()
        image = self.camera.depth_mask()
        return image
    
    def reconstruct(self, object_pose, observe_height=0.35, file_name='hull'):
        bias = np.array([-0.05, -0.03, 0.15])
        path = self.paln(((object_pose[0][0] + bias[0], object_pose[0][1] + bias[1], object_pose[0][2] + 0.2 + bias[2]), quaternion_from_euler(math.pi, 0, -math.pi/4)))
        t = Trajectory(path)
        t.control()
        center = self.camera.reconstruct(file_name) # Create the mesh file


    
    def get_joint_states(self):
        while not rospy.is_shutdown():

            joint_states = np.array(self.state_subscriber.joint_states.position)
            if len(joint_states) == 9:
                break

            print(' Waiting for joint states...')
            time.sleep(0.1)
        return joint_states


    

##################################################

class Go():
    def __init__(self) -> None:
        self.obstacles = [] # TODO
        self.state_subscriber = StateSubscriber()
        self.cmd_publisher = CmdPublisher()
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.move_group.set_planner_id('RRTConnectkConfigDefault')
    
    def paln_moveit(self, pose):
        # Initial config
        start_conf = self.get_joint_states()
        robot_state = RobotState()
        robot_state.joint_state.name = self.move_group.get_joints()[:7]
        robot_state.joint_state.position = start_conf[:7]
        self.move_group.set_start_state(robot_state)

        # Goal pose
        pose = [pose[0][0], pose[0][1], pose[0][2], pose[1][0], pose[1][1], pose[1][2], pose[1][3]]
        self.move_group.set_pose_target(pose)
        ourput = self.move_group.plan()
        success = ourput[0]
        robot_trajectory = ourput[1]

        # Save trajectory
        path = []
        if success:
            for point in robot_trajectory.joint_trajectory.points:
                path.append(point.positions)
        else:
            success = False

        return path
    
    def control(self, pose):
        
        path = self.paln_moveit(pose)
        if len(path) == 0:
            print('No path found')
            return None
        t = Trajectory(path)
        t.control()
        open_gripper()

    def get_joint_states(self):
        while not rospy.is_shutdown():
            joint_states = np.array(self.state_subscriber.joint_states.position)
            if len(joint_states) == 9:
                break
        return joint_states