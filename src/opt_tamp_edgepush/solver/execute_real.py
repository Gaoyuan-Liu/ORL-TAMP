#!/usr/bin/env python
# 
# Run this in the terminal:
# source /home/liu/catkin_ws/devel/setup.bash

import os, sys, time, math
import pybullet as pb
import rospy

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')

from examples.pybullet.utils.pybullet_tools.panda_primitives import Pose, Conf, get_ik_ir_gen, get_motion_gen, get_ik_fn,\
    get_stable_gen, get_grasp_gen, Clean, Cook, control_commands, \
    get_gripper_joints, apply_commands, State, replanner

from examples.pybullet.utils.pybullet_tools.utils import connect, get_pose, is_placement, point_from_pose, \
    disconnect, get_joint_positions, enable_gravity, save_state, restore_state, HideOutput, \
    get_distance, LockRenderer, get_min_limit, get_max_limit, has_gui, WorldSaver, wait_if_gui, add_line, SEPARATOR, create_obj

from examples.pybullet.utils.pybullet_tools.franka_primitives import StateSubscriber, CmdPublisher, GripperCommand, Attach, Detach, Trajectory, open_gripper, close_gripper, Reset, Observe
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion





from policies import EdgePushPolicyRealWorld
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
# from pushing import Pushing



def interpret_path(path):
  interpreted_path = []
  for q in path:
    interpreted_path.append(q.values)
  return interpreted_path
    


#######################################################

class Executor():
    def __init__(self, scn, s2r):
        # rospy.init_node('executor', anonymous=True)
        self.edgepush = EdgePushPolicyRealWorld(scn, s2r)
        self.sim2real = s2r

    def action2commands(self, problem, action, teleport=False):
        if action is None:
            return None

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

        elif name == 'push':
            a, b, p, lg = args
            edgepush = self.edgepush # EdgePushPolicy(scn)
            edgepush.specify(b, lg)
            commands = [edgepush]

        elif name == 'observe':
            a, o, lg = args
            # observe = Observe()
            commands = []

        else:
            raise ValueError(name)
        print(name, args, commands) 
        return commands


    #######################################################

    def apply_commands(self, commands):
        for i, command in enumerate(commands):
            command.apply()

    #######################################################

    def execute(self, problem, plan):
        # Reset
        
        reset = Reset()
        # reset.apply()
        refine_objs = {}


        for i, action in enumerate(plan):
            (name, args) = action
            if name == 'observe':
                # Get the object
                a, o, lg = args # Arm, object, loss goal
                
                dish_id = self.sim2real[o]
                observe = Observe()
                pose_dict = observe.apply([dish_id])

                # Dish
                quat = quaternion_from_euler(0, 0, -math.pi/2)
                dish_pose = pose_dict[dish_id]
                pb.removeBody(o)

                # file_path = os.path.dirname(os.path.realpath(__file__)) + '/../../camera/'
                new_o = create_obj('./hull.obj', mass=1)
                obstacle = create_obj('./hull_background.obj')
                refine_objs[o] = new_o
                pb.resetBasePositionAndOrientation(o, [dish_pose[0][0], dish_pose[0][1], 0.18], quat)
                pb.resetBasePositionAndOrientation(obstacle, [dish_pose[0][0], dish_pose[0][1], 0.18], quat)
                input('Press enter to continue: ')

            if name == 'pick':
                reset.apply()
                a, b, p, g, _, c = args
                if b in refine_objs:
                    print("\033[32m {}\033[00m" .format(' Plan Refine.'))
                    saver = WorldSaver()
                    with LockRenderer(lock=False): 
                        # Replan Grasp
                        grasp_fn = get_grasp_gen(problem, collisions=False)
                        g_list = grasp_fn(refine_objs[b])
                        saver.restore()
                        if g_list is None:
                            print("\033[32m {}\033[00m" .format(' Grasp failed'))
                        
                        # Replan Trajectory
                        pos, orn = pb.getBasePositionAndOrientation(refine_objs[b])
                        p_new = Pose(refine_objs[b], (pos, orn))

                        # obstacles = list(set(problem.fixed).difference(problem.marks))
                        obstacles = [obstacle, refine_objs[b]]
                        for g in g_list:
                            c_new = replanner(a, refine_objs[b], p_new, g[0], obstacles)
                            if c_new is not None:
                                c_new = c_new[0]
                                g_new = g[0]
                                break
                            else:
                                print("\033[32m {}\033[00m" .format(' Trajectory failed'))
                                
                        saver.restore()

                        # plan[i+1] = (next_name, (a, b, p_new, g_new, _, c_new))
                        action = ('pick', (a, refine_objs[b], p_new, g_new, _, c_new))
                    del refine_objs[b]

            commands = self.action2commands(problem, action)
            state = self.apply_commands(commands)



#######################################################

def main(verbose=True):
    rospy.init_node('executor', anonymous=True)
    reset = Reset()
    reset.apply()

if __name__ == '__main__':
    main()