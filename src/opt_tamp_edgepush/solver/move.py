
import os, sys, time
import math
import pybullet as pb
import numpy as np
from gym import spaces

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion
from examples.pybullet.utils.pybullet_tools.franka_primitives import Trajectory, StateSubscriber, CmdPublisher, open_gripper, close_gripper, Reset, Observe, Go



sys.path.insert(0, file_path + '/../../')
from camera.camera import Camera
import rospy



if __name__ == "__main__":
    rospy.init_node('move', anonymous=True)
    # pb.connect(pb.GUI)
    go = Go()
    quat = quaternion_from_euler(math.pi, 0, -math.pi/4)
    quat_c2e = quaternion_from_euler(0, 0, math.pi/2)
    T_c2e = ((0.05, -0.04, 0.05), quat_c2e) # camera to end effector   
    go.apply(((0.45, -0.04, 0.56+0.05), quat))

    # What is the bias?