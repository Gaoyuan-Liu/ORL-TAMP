from __future__ import print_function
from turtle import pos


import numpy as np
import time
import os
import sys
from matplotlib import pyplot as plt
import math

file_path = os.path.dirname(os.path.realpath(__file__))
# os.chdir(file_path + '/../../../')
# print(os.getcwd())

sys.path.insert(0, file_path + '/../../../')

# from examples.pybullet.utils.pybullet_tools.pr2_problems import create_pr2, create_table, Problem
# from examples.pybullet.utils.pybullet_tools.pr2_utils import get_other_arm, get_carry_conf, set_arm_conf, open_arm, \
#     arm_conf, REST_LEFT_ARM, close_arm, set_group_conf

from examples.pybullet.utils.pybullet_tools.utils import connect, WorldSaver, get_pose, is_placement, disconnect, \
    get_joint_positions, HideOutput, LockRenderer, wait_for_user

from examples.pybullet.utils.pybullet_tools.utils import get_bodies, sample_placement, pairwise_collision, \
    add_data_path, load_pybullet, set_point, Point, create_box, stable_z, joint_from_name, get_point, wait_for_user, set_euler, \
    RED, GREEN, BLUE, BLACK, WHITE, BROWN, TAN, GREY, joint_controller, joint_controller_hold, step_simulation, control_joints


# for panda
from examples.pybullet.utils.pybullet_tools.panda_problems import create_panda, create_table, Problem
from examples.pybullet.utils.pybullet_tools.panda_utils import get_other_arm, get_carry_conf, set_arm_conf, open_arm, \
     arm_conf, REST_LEFT_ARM, close_arm, set_group_conf, get_gripper_joints

from examples.pybullet.utils.pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, joints_from_names, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, Euler, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, plan_joint_motion, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, TAN, INF, BodySaver

from examples.pybullet.utils.pybullet_tools.panda_primitives import Conf, Trajectory, Commands, State, control_commands

from examples.pybullet.utils.pybullet_tools.ikfast.franka_panda.ik import panda_inverse_kinematics_background

import pybullet as p
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler

import moveit_commander
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import Pose


# file_path = os.path.dirname(os.path.realpath(__file__))

# sys.path.insert(0, file_path + '/../')
from common.scn import load_ee


class Pushing:
    def __init__(self, ee=None):
        if ee == None:  
            self.ee = load_ee(pos=(1,1,1))
        else:
            self.ee = ee
            for j in range(p.getNumJoints(self.ee)):
                info = p.getJointInfo(ee, j)
                jointType = info[2]
                if (jointType == p.JOINT_PRISMATIC): # 9, 10
                    p.resetJointState(ee, j, 0.0) 


        self.away_pose = ((1,1,1),(0,0,0,1))
        # Moveit
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.move_group.set_planner_id('RRTkConfigDefault')

    
    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        # print(f'vector = {vector}')
        if np.linalg.norm(vector) == 0:
            return vector
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))





    def push(self, robot, start_point, end_point, obstacles=[], sim_execute=True, initial_conf = (0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0)):
        
        # Push
        # self.push_core(robot, move_group, start, end, obj_poses) # [0.6, 0.2], [0.6, -0.1]

        joints = get_movable_joints(robot)
        arm_joints = joints[:7]
        finger_joints = joints[7:]
        push_height = 0.13 + 0.15

        # Calculate orientation
        yaw = self.angle_between([1,0], np.subtract(end_point, start_point))
        if start_point[1] > end_point[1]:
            yaw = -yaw

        # Position end
        quat = quaternion_from_euler(0, 3.14, 0, axes='sxyz')
        init_pose = Pose()
        init_pose.position.x = 0.4
        init_pose.position.y = 0.0
        init_pose.position.z = 0.7
        init_pose.orientation.x = quat[0]
        init_pose.orientation.y = quat[1]
        init_pose.orientation.z = quat[2]
        init_pose.orientation.w = quat[3]

        # Position start
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        start_pose = [np.append(start_point, push_height), quaternion_from_euler(0, 3.14, yaw, axes='sxyz')]
        # start_conf = panda_inverse_kinematics_background(robot, 'panda', start_pose)

        # IK checking
        # if start_conf == None:
        #     print(' No IK solution.')
            # return
        
        # EE Collsion checking
        saver = WorldSaver()
        with LockRenderer(lock=True):
            # p.setRealTimeSimulation(0)
            p.resetBasePositionAndOrientation(self.ee, start_pose[0], start_pose[1])
            p.performCollisionDetection()
            for i in obstacles:
                contact_points = p.getContactPoints(self.ee, i)
                if len(contact_points) > 0:
                    collision = True
                    print(' Start pose collision.')
                    # time.sleep(0.5)
                    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)    
                    p.resetBasePositionAndOrientation(self.ee, self.away_pose[0], self.away_pose[1])
                    return
            saver.restore()
                
        
        p.resetBasePositionAndOrientation(self.ee, self.away_pose[0], self.away_pose[1])
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        

        # Length checking
        # if len(start_conf) > 7:
        #     start_conf = start_conf[:7]

        # robot_state = RobotState()
        # robot_state.joint_state.name = move_group.get_joints()[:7]
        # robot_state.joint_state.position = start_conf

        # set_joint_positions(robot, arm_joints, start_conf)
        # move_group.set_start_state(robot_state)

        quat = quaternion_from_euler(0, 3.14, yaw-3.14/4, axes='sxyz')
        start_pose = Pose()
        start_pose.position.x = start_point[0] #(end_point[0] + start_point[0])/2
        start_pose.position.y = start_point[1] #(end_point[1] + start_point[1])/2
        start_pose.position.z = push_height
        start_pose.orientation.x = quat[0]
        start_pose.orientation.y = quat[1]
        start_pose.orientation.z = quat[2]
        start_pose.orientation.w = quat[3]

        # Position end
        quat = quaternion_from_euler(0, 3.14, yaw-3.14/4, axes='sxyz')
        end_pose = Pose()
        end_pose.position.x = end_point[0]
        end_pose.position.y = end_point[1]
        end_pose.position.z = push_height
        end_pose.orientation.x = quat[0]
        end_pose.orientation.y = quat[1]
        end_pose.orientation.z = quat[2]
        end_pose.orientation.w = quat[3]

        # Set Cartesian path
        robot_state = RobotState()
        robot_state.joint_state.name = self.move_group.get_joints()[:7]
        robot_state.joint_state.position = initial_conf
        self.move_group.set_start_state(robot_state)

        waypoints = [start_pose, end_pose]
        (plan_waypoints, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step = 0.004, jump_threshold=5.0)

        if fraction < 0.8:
            # print('Invalid yaw angle, replan.')
            quat = quaternion_from_euler(0, 3.14, yaw+3.14/4*3, axes='sxyz')
            start_pose.orientation.x = quat[0]
            start_pose.orientation.y = quat[1]
            start_pose.orientation.z = quat[2]
            start_pose.orientation.w = quat[3]
            end_pose.orientation.x = quat[0]
            end_pose.orientation.y = quat[1]
            end_pose.orientation.z = quat[2]
            end_pose.orientation.w = quat[3]

            waypoints = [start_pose, end_pose]
            (plan_waypoints, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step = 0.004, jump_threshold=5.0)
        
        ######################################
        if sim_execute == True:
            if fraction > 0.8:
                # p.setRealTimeSimulation(1)
                for point in plan_waypoints.joint_trajectory.points:
                    control_joints(robot, arm_joints, point.positions)
                    control_joints(robot, finger_joints, [0.0,0.0])
                    p.stepSimulation()
                    time.sleep(0.0075)

                for point in plan_waypoints.joint_trajectory.points[::-1]:
                    control_joints(robot, arm_joints, point.positions)
                    control_joints(robot, finger_joints, [0.0,0.0])
                    p.stepSimulation()
                    time.sleep(0.005)

                    
                 
                time.sleep(0.5) # Wait for the object to stop moving
                # p.setRealTimeSimulation(0)
                return
        ######################################
            print("\033[91m {}\033[00m" .format(f'\n Pushing Failed'))
        else:

            # https://docs.ros.org/en/noetic/api/trajectory_msgs/html/msg/JointTrajectory.html
            return plan_waypoints.joint_trajectory.points

        return
      

    def push_plan(self, start_point, end_point, obstacles=[], initial_conf = (0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0), push_height=0.12+0.17):
        
        # Push
        push_height = push_height

        # Calculate orientation
        yaw = self.angle_between([1,0], np.subtract(end_point, start_point))
        if start_point[1] > end_point[1]:
            yaw = -yaw



        # Position start
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        start_pose = [np.append(start_point, push_height), quaternion_from_euler(0, 3.14, yaw, axes='sxyz')]
        # start_conf = panda_inverse_kinematics_background(robot, 'panda', start_pose)

        # EE Collsion checking
        # saver = WorldSaver()
        # with LockRenderer(lock=True):
        #     # p.setRealTimeSimulation(0)
        #     p.resetBasePositionAndOrientation(self.ee, start_pose[0], start_pose[1])
        #     p.performCollisionDetection()
        #     for i in obstacles:
        #         contact_points = p.getContactPoints(self.ee, i)
        #         if len(contact_points) > 0:
        #             collision = True
        #             print(' Start pose collision.')
        #             # time.sleep(0.5)
        #             # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)    
        #             p.resetBasePositionAndOrientation(self.ee, self.away_pose[0], self.away_pose[1])
        #             return
        #     saver.restore()
                
        
        p.resetBasePositionAndOrientation(self.ee, self.away_pose[0], self.away_pose[1])

        
        quat = quaternion_from_euler(0, 3.14, yaw-3.14/4, axes='sxyz')
        start_pose = Pose()
        start_pose.position.x = start_point[0] #(end_point[0] + start_point[0])/2
        start_pose.position.y = start_point[1] #(end_point[1] + start_point[1])/2
        start_pose.position.z = push_height 
        start_pose.orientation.x = quat[0]
        start_pose.orientation.y = quat[1]
        start_pose.orientation.z = quat[2]
        start_pose.orientation.w = quat[3]

        # Position end
        quat = quaternion_from_euler(0, 3.14, yaw-3.14/4, axes='sxyz')
        end_pose = Pose()
        end_pose.position.x = end_point[0]
        end_pose.position.y = end_point[1]
        end_pose.position.z = push_height
        end_pose.orientation.x = quat[0]
        end_pose.orientation.y = quat[1]
        end_pose.orientation.z = quat[2]
        end_pose.orientation.w = quat[3]

        # Set Cartesian path
        robot_state = RobotState()
        robot_state.joint_state.name = self.move_group.get_joints()[:7]
        robot_state.joint_state.position = initial_conf[:7]
        self.move_group.set_start_state(robot_state)

        waypoints = [start_pose, end_pose]
        (plan_waypoints, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step = 0.001, jump_threshold=5.0)

        if fraction < 0.8:
            # print('Invalid yaw angle, replan.')
            quat = quaternion_from_euler(0, 3.14, yaw+3.14/4*3, axes='sxyz')
            start_pose.orientation.x = quat[0]
            start_pose.orientation.y = quat[1]
            start_pose.orientation.z = quat[2]
            start_pose.orientation.w = quat[3]
            end_pose.orientation.x = quat[0]
            end_pose.orientation.y = quat[1]
            end_pose.orientation.z = quat[2]
            end_pose.orientation.w = quat[3]

            waypoints = [start_pose, end_pose]
            (plan_waypoints, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step = 0.001, jump_threshold=5.0)
        
        
        ######################################

        if fraction > 0.8:
            return plan_waypoints.joint_trajectory.points
        else:
            print("\033[91m {}\033[00m" .format(f'\n Pushing Failed'))
            # https://docs.ros.org/en/noetic/api/trajectory_msgs/html/msg/JointTrajectory.html
            return None #plan_waypoints.joint_trajectory.points




# if __name__ == '__main__':
    
























