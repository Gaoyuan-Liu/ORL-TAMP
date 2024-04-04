from __future__ import print_function
from turtle import pos


import numpy as np
import time
import os
import sys
from matplotlib import pyplot as plt
import math
import pybullet as pb
from tracikpy import TracIKSolver

file_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, file_path + '/../')
from utils.pybullet_tools.utils import WorldSaver, get_movable_joints, LockRenderer, Pose, control_joints
from utils.pybullet_tools.transformations import quaternion_from_euler, quaternion_to_se3
from utils.scn import load_ee


class Pushing:
    def __init__(self, ee=None):
        if ee == None:  
            self.ee = load_ee(pos=(1,1,1))
        else:
            self.ee = ee
            for j in range(pb.getNumJoints(self.ee)):
                info = pb.getJointInfo(ee, j)
                jointType = info[2]
                if (jointType == pb.JOINT_PRISMATIC): # 9, 10
                    pb.resetJointState(ee, j, 0.0) 


        self.away_pose = ((1,1,1),(0,0,0,1))
        self.ik_solver = TracIKSolver(
            dir + "/../utils/models/franka_panda/panda_modified.urdf",
                "panda_link0",
                "panda_hand",
            )
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0.0175, -2.8973])
        qinit = (q_max + q_min) / 2


    def panda_tracik_fn(robot, pose, qinit=None):
        ee_pose = quaternion_to_se3(pose[1], pose[0])
        qout = ik_solver.ik(ee_pose, qinit=qinit)
        return qout

    
    



    def cartesian_planner(self, start_pose, end_pose):

    def push(self, robot, start_point, end_point, obstacles=[], sim_execute=True, initial_conf = (0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0)):
        
        joints = get_movable_joints(robot)
        arm_joints = joints[:7]
        finger_joints = joints[7:]

        push_height = 0.13 + 0.15

        # Calculate orientation
        yaw = self.angle_between([1,0], np.subtract(end_point, start_point))
        if start_point[1] > end_point[1]:
            yaw = -yaw

        

        # Position start
        start_pose = [np.append(start_point, push_height), quaternion_from_euler(0, 3.14, yaw, axes='sxyz')]

        # EE Collsion checking
        saver = WorldSaver()
        with LockRenderer(lock=False):
            # pb.setRealTimeSimulation(0)
            pb.resetBasePositionAndOrientation(self.ee, start_pose[0], start_pose[1])
            # input('Press enter to continue: ')
            pb.performCollisionDetection()
            for i in obstacles:
                contact_points = pb.getContactPoints(self.ee, i)
                if len(contact_points) > 0:
                    collision = True
                    print(' Start pose collision.')
                    # time.sleep(0.5)
                    # pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)    
                    pb.resetBasePositionAndOrientation(self.ee, self.away_pose[0], self.away_pose[1])
                    return
            saver.restore()
        pb.resetBasePositionAndOrientation(self.ee, self.away_pose[0], self.away_pose[1])

        
        
        
        ######################################
        if sim_execute == True:
            if fraction > 0.8:
                # pb.setRealTimeSimulation(1)
                for point in plan_waypoints.joint_trajectory.points:
                    control_joints(robot, arm_joints, point.positions)
                    control_joints(robot, finger_joints, [0.0,0.0])
                    pb.stepSimulation()
                    time.sleep(0.0075)

                for point in plan_waypoints.joint_trajectory.points[::-1]:
                    control_joints(robot, arm_joints, point.positions)
                    control_joints(robot, finger_joints, [0.0,0.0])
                    pb.stepSimulation()
                    time.sleep(0.005)

                    
                 
                time.sleep(0.5) # Wait for the object to stop moving
                # pb.setRealTimeSimulation(0)
                return
        ######################################
            print("\033[91m {}\033[00m" .format(f'\n Pushing Failed'))
        else:

            # https://docs.ros.org/en/noetic/api/trajectory_msgs/html/msg/JointTrajectory.html
            return plan_waypoints.joint_trajectory.points

        return
    

    ####################################################3

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
      

########################################################3
    

