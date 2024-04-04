


import os, sys, math
import pybullet as pb
import numpy as np
from tracikpy import TracIKSolver

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
from utils.pybullet_tools.transformations import quaternion_to_se3

pandaNumDofs = 7
PANDA_ARM_JOINTS_ID = range(pandaNumDofs)
PANDA_FINGERS__ID = [9, 10]
pandaEndEffectorIndex = 11


class Control:
    def __init__(self) -> None:
  
        # IK constraints
        self.ll = [-7]*pandaNumDofs
        # Upper limits for null space (todo: set them to proper range)
        self.ul = [7]*pandaNumDofs
        # Joint ranges for null space (todo: set them to proper range)
        self.jr = [7]*pandaNumDofs
        # Restposes for null space
        self.jointResetPositions=[0.0, 0.0, 0.0, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
        self.rp = self.jointResetPositions

        # IK solver
        dir = os.path.dirname(os.path.realpath(__file__))
        self.ik_solver = TracIKSolver(
            dir + "/../utils/models/franka_panda/panda_modified.urdf",
            "panda_link0",
            "panda_hand",
        )

        q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0.0175, -2.8973])
        self.qinit = (q_max + q_min) / 2

    def panda_tracik_fn(self, robot, pose, qinit=None):
        ee_pose = quaternion_to_se3(pose[1], pose[0])
        if qinit is None:
            qinit = self.qinit
        qout = self.ik_solver.ik(ee_pose, qinit=qinit)
        return qout


    def go_joint_space(self, robot, jointPoses):
        for i in range(pandaNumDofs):
            pb.setJointMotorControl2(robot, i, pb.POSITION_CONTROL, jointPoses[i],force=5 * 240.)


    def go_cartesian_space(self, robot, eePose):
        # IK
        jointPoses = pb.calculateInverseKinematics(robot, pandaEndEffectorIndex, eePose[0], eePose[1], self.ll, self.ul,
        self.jr, self.rp, maxNumIterations=20)
        # Go
        self.go_joint_space(robot, jointPoses)


    def finger_close(self, robot):
        finger_target = 0.00
        for i in [9,10]:
            pb.setJointMotorControl2(robot, i, pb.POSITION_CONTROL,finger_target ,force= 10)


    def finger_open(self, robot):
        finger_target = 0.04
        for i in [9,10]:
            pb.setJointMotorControl2(robot, i, pb.POSITION_CONTROL,finger_target ,force= 10)

    


    def set_joint_space(self, robot, jointPoses, fingers=None):
        index = 0
        # for j in range(pb.getNumJoints(robot)):
        for j in range(7):
            pb.changeDynamics(robot, j, linearDamping=0, angularDamping=0)
            info = pb.getJointInfo(robot, j)
            jointName = info[1]
            jointType = info[2]
            # if (jointType == pb.JOINT_PRISMATIC): # 9, 10
            #     pb.resetJointState(robot, j, jointPoses[index]) 
            #     index=index+1
            if (jointType == pb.JOINT_REVOLUTE):
                pb.resetJointState(robot, j, jointPoses[index]) 
                index=index+1
        
        for i in [9,10]:
            if fingers == 'open':
                pb.resetJointState(robot, i, 0.03)
            elif fingers == 'close':
                pb.resetJointState(robot, i, 0.00)
            else:
                pass

            


    def set_cartesian_space(self, robot, eePose, fingers=None):
        # IK
        jointPoses = self.panda_tracik_fn(robot, eePose)
        # Go
        if jointPoses is not None:
            self.set_joint_space(robot, jointPoses, fingers=fingers)
        else:
            print('No IK solution found.')


        
        