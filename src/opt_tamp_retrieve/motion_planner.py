
import numpy as np
import time
import os
import sys
import math
import pybullet as p

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../../')

import rospy
import moveit_commander
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import Pose



class CartesianPlanner:
    def __init__(self):
        # self.collision_checker = Shadow()
        # rospy.init_node('cartesian_planner')
        # Moveit
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.move_group.set_planner_id('RRTConnectkConfigDefault')
    

    # ---------------------------------------------------

    def plan(self, start_conf, end_pose, eef_step=0.008):

        robot_state = RobotState()
        robot_state.joint_state.name = self.move_group.get_joints()[:7]
        robot_state.joint_state.position = start_conf[:7]
        self.move_group.set_start_state(robot_state)

        # Formulize end pose
        end_pose_msg = Pose()
        end_pose_msg.position.x = end_pose[0][0]
        end_pose_msg.position.y = end_pose[0][1]
        end_pose_msg.position.z = end_pose[0][2]
        end_pose_msg.orientation.x = end_pose[1][0]
        end_pose_msg.orientation.y = end_pose[1][1]
        end_pose_msg.orientation.z = end_pose[1][2]
        end_pose_msg.orientation.w = end_pose[1][3]

        # Set Cartesian path
        waypoints = [end_pose_msg]
        (plan_waypoints, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step, jump_threshold=5.0)


        # Save trajectory
        trajectory = []
        for point in plan_waypoints.joint_trajectory.points:
            trajectory.append(point.positions)

        return trajectory
    
    # --------------------------------------------------- 

    def plan_multipose(self, start_conf, end_pose_list, eef_step=0.008):

        robot_state = RobotState()
        robot_state.joint_state.name = self.move_group.get_joints()[:7]
        robot_state.joint_state.position = start_conf[:7]
        self.move_group.set_start_state(robot_state)

        # Formulize end pose
        waypoints = []
        for i in range(len(end_pose_list)):
            end_pose_msg = Pose()
            end_pose_msg.position.x = end_pose_list[i][0][0]
            end_pose_msg.position.y = end_pose_list[i][0][1]
            end_pose_msg.position.z = end_pose_list[i][0][2]
            end_pose_msg.orientation.x = end_pose_list[i][1][0]
            end_pose_msg.orientation.y = end_pose_list[i][1][1]
            end_pose_msg.orientation.z = end_pose_list[i][1][2]
            end_pose_msg.orientation.w = end_pose_list[i][1][3]

            # Set Cartesian path
            waypoints.append(end_pose_msg)

        (plan_waypoints, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step, jump_threshold=5.0)

        # print("fraction: ", fraction)



        # Save trajectory
        trajectory = []
        if fraction > 0.7:
            success = True
            for point in plan_waypoints.joint_trajectory.points:
                trajectory.append(point.positions)
        else:
            success = False


        return success, trajectory

    # ---------------------------------------------------

    def plan_joint(self, start_conf, end_config):
        robot_state = RobotState()
        robot_state.joint_state.name = self.move_group.get_joints()[:7]
        robot_state.joint_state.position = start_conf[:7]
        self.move_group.set_start_state(robot_state)

        self.move_group.set_joint_value_target(end_config)
        plan_tuple = self.move_group.plan()
        plan = plan_tuple[1]
        trajectory = []
        for point in plan.joint_trajectory.points:
            trajectory.append(point.positions)

        return trajectory




    def plan2(self, start_conf, start_pose, end_pose):

        robot_state = RobotState()
        robot_state.joint_state.name = self.move_group.get_joints()[:7]
        robot_state.joint_state.position = start_conf
        self.move_group.set_start_state(robot_state)

        # Formulize start pose
        start_pose_msg = Pose()
        start_pose_msg.position.x = start_pose[0][0]
        start_pose_msg.position.y = start_pose[0][1] 
        start_pose_msg.position.z = start_pose[0][2]
        start_pose_msg.orientation.x = start_pose[1][0]
        start_pose_msg.orientation.y = start_pose[1][1]
        start_pose_msg.orientation.z = start_pose[1][2]
        start_pose_msg.orientation.w = start_pose[1][3]

        # Formulize end pose
        end_pose_msg = Pose()
        end_pose_msg.position.x = end_pose[0][0]
        end_pose_msg.position.y = end_pose[0][1]
        end_pose_msg.position.z = end_pose[0][2]
        end_pose_msg.orientation.x = end_pose[1][0]
        end_pose_msg.orientation.y = end_pose[1][1]
        end_pose_msg.orientation.z = end_pose[1][2]
        end_pose_msg.orientation.w = end_pose[1][3]

        # Set Cartesian path
        waypoints = [end_pose_msg]
        (plan_waypoints, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step = 0.008, jump_threshold=4.0)



        # Save trajectory
        trajectory = []
        for point in plan_waypoints.joint_trajectory.points:
            # control_joints(robot, arm_joints, point.positions)
            # control_joints(robot, finger_joints, [0.0,0.0])
            # time.sleep(0.01)
            trajectory.append(point.positions)
            # for _ in joint_controller_hold(robot, arm_joints, point.positions, timeout=1):
            #     step_simulation()

        return trajectory



if __name__ == '__main__':
    # rospy.init_node('cartesian_planner')

    move_group = moveit_commander.MoveGroupCommander("panda_arm")

    # robot_state = RobotState()
    # robot_state.joint_state.name = move_group.get_joints()[:7]
    # robot_state.joint_state.position = [1.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0]
    # move_group.set_start_state(robot_state)

    start_conf = [0.0, 0.0, 0.0, -1.0, 0.0, 1.5, 0.0]
    start_pose = [[0.4,-0.1, 0.4],[-0.707107, 0.0, 0.0, 0.707107]]
    end_pose = [[0.4,0.1, 0.3],[-0.707107, 0.0, 0.0, 0.707107]]

    planner = CartesianPlanner()
    t = planner.plan(start_conf, end_pose)

    print(len(t))


