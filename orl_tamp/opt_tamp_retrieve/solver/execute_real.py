#!/usr/bin/env python

# Run this in the terminal:
# source /home/liu/catkin_ws/devel/setup.bash


import os, sys, time, math
import rospy
#import gym
import numpy as np
import pybullet as pb
from scipy.interpolate import interp1d

# ROS
from franka_gripper.msg import MoveAction, GraspEpsilon, MoveGoal, GraspAction, GraspGoal, GraspActionGoal
import actionlib
from std_msgs.msg import Float64, Int64, Float64MultiArray
from sensor_msgs.msg import JointState

# 
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../')
# sys.path.insert(0, file_path + '/../pddlstream/examples/pybullet/utils/')
sys.path.insert(0, file_path + '/../pddlstream/')

from examples.pybullet.utils.pybullet_tools.panda_primitives import BodyConf, CommandMP, get_simple_ik_fn, get_free_motion_gen, get_grasp_gen, Pose, replanner
from examples.pybullet.utils.pybullet_tools.franka_primitives import StateSubscriber, CmdPublisher, GripperCommand, Attach, Detach, Trajectory, open_gripper, close_gripper, Reset, Observe
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion

from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, LockRenderer
from common.scn import Scenario 
from camera.camera import Camera
from policies import HookingPolicyRealWorld

#############################################
# Motion Planner

# 1. Get from our planner
# 2. Get from moveit

def motion_planner(start_config, end_pose, start_pose=None, obstacles=[], teleport=False): # The goal pose shold be in the world space
  # Build pybullet env
  
  scn = Scenario()
  obstacles.append(scn.floor)
  
  robot = scn.add_robot(pose=((0,0,0), (0,0,0,1))) 
  saved_world = WorldSaver()
  # Functions
  ik_fn = get_simple_ik_fn(robot, fixed=obstacles, teleport=teleport)
  free_motion_fn = get_free_motion_gen(robot, fixed=obstacles, teleport=teleport, algorithm='rrt')

  # Start config
  if start_pose == None:
    conf0 = BodyConf(robot, configuration=start_config)
  else:
    conf0 = ik_fn(pose=start_pose)
  
  # End config
  conf1 = ik_fn(pose=end_pose)

  if conf1 == None:
      print('No IK found.')
      saved_world.restore()
      return 

  # Plan
  result = free_motion_fn(conf0, conf1)
  if result == None:
    saved_world.restore()
    return None
  path, = result
  saved_world.restore()
  return CommandMP(path.body_paths)

#############################################

def motion_planner_moveit(): #TODO
  pass

#############################################

# Franka Primitives



def interpret_path(path):
  interpreted_path = []
  for q in path:
    interpreted_path.append(q.values)
  return interpreted_path


#############################################
class Executor():
  def __init__(self,scn):
    rospy.init_node('executor', anonymous=True)
    self.cmd_publisher = CmdPublisher()
    self.state_subscriber = StateSubscriber()

    self.hook = HookingPolicyRealWorld()
    self.scn = scn

    self.sim2real = {2:'0', 3:'2', 4:'5'}

  def path_execute(self, path, duration=4,control_freq=100):
    # Make trajectory
    time_vector = np.linspace(0, duration, len(path)) # (start_time, end_time(s), len(waypoints))
    joint_trajectory = interp1d(time_vector, path, axis=0, kind='cubic')

    # Loop
    rate = rospy.Rate(control_freq)

    start_time = rospy.get_time()
    while not rospy.is_shutdown():
      current_time = rospy.get_time() - start_time
      if current_time > duration:
        break
      current_waypoint = joint_trajectory(current_time)
      # joint_msg = Float64MultiArray(data=current_waypoint)
      self.cmd_publisher.publish(current_waypoint[:7])

      # end_time = time.time()
      # d_time = end_time - start_time
      rate.sleep()


  def action2commands(self, problem, action, teleport=False):
      if action is None:
          return None
      # commands = []
      (name, args) = action 

      print(f'\nname = {name}\n') 
  
      if name == 'pick':
          a, b, p, g, _, c = args
          [t] = c.commands
          path = t.path
          path = interpret_path(path)
          t_franka = Trajectory(path)
          gripper_mode = 'close'
          close_gripper = GripperCommand(problem.robot, a, gripper_mode, teleport=teleport)
          attach = Attach(problem.robot, a, g, b)
          commands = [t_franka, close_gripper, attach, t_franka.reverse()]

      elif name == 'place':
          a, b, p, g, _, c = args
          [t] = c.commands
          path = t.path
          path = interpret_path(path)
          t_franka = Trajectory(path)
          gripper_mode = 'open'
          open_gripper = GripperCommand(problem.robot, a, gripper_mode, teleport=teleport)
          detach = Detach(problem.robot, a, b)
          commands = [t_franka, detach, open_gripper, t_franka.reverse()]

      elif name == 'hook':
          a, b, p, lg = args
          self.hook.specify(b, lg)
          commands = [self.hook]

      elif name == 'observe':
          a, b, lg = args
          # observe = Observe()
          # commands = [observe]
          commands = []

      else:
          raise ValueError(name)
      print(name, args, commands) 
      return commands

  def apply_commands(self, commands):
    for i, command in enumerate(commands):
       command.apply()
      
  

  def execute(self, problem, plan):
    # Reset
    reset = Reset()
    reset.apply()

    refine_objs = []
    for i, action in enumerate(plan):
      (name, args) = action
      if name == 'observe':
          
          # Get the object
          a, o, lg = args # Arm, object, loss goal
          refine_objs.append(o)

          cup_id = self.sim2real[o]
          observe = Observe()
          pose_dict = observe.apply([cup_id])

          # Cup
          cup_pose = pose_dict[cup_id]
          pb.resetBasePositionAndOrientation(o, [cup_pose[0][0], cup_pose[0][1], 0.03], cup_pose[1])
          # Hook
          # hook_id = '1'
          # hook_pose = pose_dict[hook_id]
          # hook_pose = ((1,1,1),hook_pose[1])
          # pb.resetBasePositionAndOrientation(self.scn.hook, hook_pose[0], hook_pose[1])

          pb.setRealTimeSimulation(1)
          time.sleep(2)
          pb.setRealTimeSimulation(0)
          reset.apply()
          
          # ATTENTION: This will cause noise

      if name == 'pick':
          a, b, p, g, _, c = args
          if b in refine_objs:
              print("\033[32m {}\033[00m" .format(' Plan Refine.'))
              saver = WorldSaver()
              with LockRenderer(lock=False): 
                  # Replan Grasp
                  grasp_fn = get_grasp_gen(problem, collisions=False)
                  g_list = grasp_fn(b)
                  saver.restore()
                  if g_list is None:
                      print("\033[32m {}\033[00m" .format(' Grasp failed'))
                  
                  # Replan Trajectory
                  pos, orn = pb.getBasePositionAndOrientation(b)
                  p_new = Pose(b, (pos, orn))

                  obstacles = list(set(problem.fixed).difference(problem.marks))
                  for g in g_list:
                      c_new = replanner(a, b, p_new, g[0], obstacles)
                      if c_new is not None:
                          c_new = c_new[0]
                          g_new = g[0]
                          break
                      else:
                          print("\033[32m {}\033[00m" .format(' Trajectory failed'))
                          

                  saver.restore()

                  # plan[i+1] = (next_name, (a, b, p_new, g_new, _, c_new))
                  action = ('pick', (a, b, p_new, g_new, _, c_new))
                  refine_objs.remove(b)


    # for i, action in enumerate(plan):
      commands = self.action2commands(problem, action)
      self.apply_commands(commands)




if __name__ == "__main__":
   pass

  #############################################












