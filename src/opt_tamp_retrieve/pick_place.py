#! /venv3.8/bin/python
import os, sys, math, time
import moveit_commander
import pybullet as pb
import numpy as np

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../pddlstream/')
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion
from examples.pybullet.utils.pybullet_tools.utils import connect, wait_for_user

from control import Control
from env_flexible import RetrieveEnv
sys.path.insert(0, file_path + '/../')
from common.scn import Scenario
from motion_planner import CartesianPlanner


def pick_place(robot, tool, object_goal_pose, control, cartesian_planner):

    move_group = moveit_commander.MoveGroupCommander("panda_arm")
    move_group.set_planner_id('RRTConnectkConfigDefault')

    # Go approach
    info = pb.getBasePositionAndOrientation(tool)
    ee_pos1 = list(info[0])
    ee_pos1[-1] = ee_pos1[-1] + 0.14
    tool_orn = info[1]
    hook_rpy = euler_from_quaternion(tool_orn)
    ee_yaw1 = (hook_rpy[-1] - math.pi/4) % (np.sign(hook_rpy[-1] - math.pi/4)*(math.pi))
    ee_orn1 = quaternion_from_euler(math.pi, 0, ee_yaw1)
    approach_pose = (ee_pos1, ee_orn1)

    # Go grasp
    ee_pos2 = ee_pos1
    ee_pos2[-1] = ee_pos1[-1] - 0.04
    ee_orn2 = ee_orn1
    grasp_pose = (ee_pos2, ee_orn2)

    # Plan and execute
    pose_list = [approach_pose, grasp_pose]
    pandaNumDofs = 7
    joint_states = pb.getJointStates(robot, list(range(pandaNumDofs)))# Given by Holding
    start_conf = []
    for i in joint_states:
        start_conf.append(i[0])
    success, trajectory = cartesian_planner.plan_multipose(start_conf, pose_list)
    for waypoint in trajectory:
        time.sleep(0.01) # Moving speed
        control.go_joint_space(robot, waypoint)
    control.finger_close(robot)
    

    # Go approach
    joint_states = pb.getJointStates(robot, list(range(pandaNumDofs)))# Given by Holding
    start_conf = []
    for i in joint_states:
        start_conf.append(i[0])
    pose_list = [approach_pose]
    success, trajectory = cartesian_planner.plan_multipose(start_conf, pose_list)
    for waypoint in trajectory:
        time.sleep(0.01) # Moving speed
        control.go_joint_space(robot, waypoint)

    # ---------------------------------------------------

    # Go goal approach 
    # Object pose --> ee pose
    ee_pos3 = list(object_goal_pose[0])
    ee_pos3[-1] = ee_pos3[-1] + 0.14 + 0.2 
    tool_orn = object_goal_pose[1]
    hook_rpy = euler_from_quaternion(tool_orn)
    ee_yaw3 = (hook_rpy[-1] - math.pi/4) % (np.sign(hook_rpy[-1] - math.pi/4)*(math.pi))
    ee_orn3 = quaternion_from_euler(math.pi, 0, ee_yaw3)
    goal_approach_pose = (ee_pos3, ee_orn3)



    joint_states = pb.getJointStates(robot, list(range(pandaNumDofs)))# Given by Holding
    start_conf = []
    for i in joint_states:
        start_conf.append(i[0])
    pose_list = [goal_approach_pose]
    success, trajectory = cartesian_planner.plan_multipose(start_conf, pose_list)
    for waypoint in trajectory:
        time.sleep(0.1) 
        control.go_joint_space(robot, waypoint)





    ee_pos4 = ee_pos3
    ee_pos4[-1] = ee_pos4[-1] - 0.2
    ee_orn4 = ee_orn3
    goal_grasp_pose = (ee_pos4, ee_orn4)

    joint_states = pb.getJointStates(robot, list(range(pandaNumDofs)))# Given by Holding
    start_conf = []
    for i in joint_states:
        start_conf.append(i[0])
    pose_list = [goal_grasp_pose]
    success, trajectory = cartesian_planner.plan_multipose(start_conf, pose_list)
    for waypoint in trajectory:
        time.sleep(0.1) 
        control.go_joint_space(robot, waypoint)
    control.finger_open(robot)



if __name__ == '__main__':
    control = Control()
    cartesian_planner = CartesianPlanner()

    # Load scene
    env = RetrieveEnv()
    pb.resetBasePositionAndOrientation(env.tool, (0.5, 0.0, 0.17), (0,0,0,1))

    pb.setRealTimeSimulation(1)
    time.sleep(0.5)


    quat = quaternion_from_euler(0, 0, -math.pi/2)
    goal_pose = ((0.5, 0.1, 0.17), quat)

    pick_place(env.robot, env.tool, goal_pose, control, cartesian_planner)
    

    input("Press Enter to continue...")
    







