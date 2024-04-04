#!/usr/bin/env python
# 
# Run this in the terminal:
# source /home/liu/catkin_ws/devel/setup.bash

import os, sys, time, math
import pybullet as pb
import rospy

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')

from examples.pybullet.utils.pybullet_tools.panda_primitives import Pose, get_grasp_gen, replanner

from examples.pybullet.utils.pybullet_tools.utils import LockRenderer, WorldSaver, create_obj

from examples.pybullet.utils.pybullet_tools.franka_primitives import StateSubscriber, CmdPublisher, GripperCommand, Attach, Detach, Trajectory, open_gripper, close_gripper, Reset, Observe
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion




file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../')

from opt_tamp_retrieve.solver.policies import RetrievePolicyRealWorld
from opt_tamp_edgepush.solver.policies import EdgePushPolicyRealWorld



def interpret_path(path):
  interpreted_path = []
  for q in path:
    interpreted_path.append(q.values)
  return interpreted_path
    


#######################################################

class Executor():
    def __init__(self, scn, s2r):
        
        self.s2r = s2r

        self.retrieve = RetrievePolicyRealWorld(s2r)
        self.edgepush = EdgePushPolicyRealWorld(s2r)
        

    def action2commands(self, problem, action, teleport=False):
        if action is None:
            return None

        (name, args) = action 

        print(f'\nname = {name}\n') 

        if name == 'pick':
            a, b, p, g, c = args
            [t] = c.commands
            path = t.path
            path = interpret_path(path)
            t_franka = Trajectory(path)
            gripper_mode = 'open'
            open_gripper = GripperCommand(problem.robot, a, gripper_mode, teleport=teleport)
            gripper_mode = 'close'
            close_gripper = GripperCommand(problem.robot, a, gripper_mode, teleport=teleport)

            attach = Attach(problem.robot, a, g, b)
            commands = [t_franka, close_gripper, attach, t_franka.reverse()]

        elif name == 'place':
            a, b, p, g, c = args
            [t] = c.commands
            path = t.path
            path = interpret_path(path)
            t_franka = Trajectory(path)
            gripper_mode = 'open'
            open_gripper = GripperCommand(problem.robot, a, gripper_mode, teleport=teleport)
            detach = Detach(problem.robot, a, b)
            commands = [t_franka, detach, open_gripper, t_franka.reverse()]

        elif name == 'retrieve':
            a, b, p, lg = args
            bar = problem.tools[0]
            self.retrieve.specify(b, lg, bar)
            commands = [self.retrieve]

        elif name == 'push':
            a, b, p, lg = args
            edgepush = self.edgepush # EdgePushPolicy(scn)
            edgepush.specify(b, lg)
            commands = [edgepush]

        elif name == 'observe_push':
            a, o, lg = args
            # observe = Observe()
            commands = []

        elif name == 'observe_retrieve':
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
            command.control()

    #######################################################

    def execute(self, problem, plan):
        # Reset
        
        reset = Reset()
        # reset.control()
        refine_objs = {}

        obstacles = []

        for i, action in enumerate(plan):
            (name, args) = action

            if name == 'observe_push':
                # Get the object
                a, o, lg = args # Arm, object, loss goal
                
                dish_id = self.s2r[o]
                observe = Observe()
                pose_dict = observe.control([dish_id])

                # Dish
                quat = quaternion_from_euler(0, 0, -math.pi/2)
                dish_pose = pose_dict[dish_id]
                pb.removeBody(o)

              
                new_o = create_obj('./hull_final.obj', mass=1)
                background = create_obj('./hull_background_final.obj')

                refine_objs[o] = new_o
                pb.resetBasePositionAndOrientation(o, [dish_pose[0][0], dish_pose[0][1], 0.2], quat)
                pb.resetBasePositionAndOrientation(background, [dish_pose[0][0], dish_pose[0][1], 0.2], quat)
                obstacles.append(background)


            if name == 'observe_retrieve':
                # Get the object
                a, o, lg = args # Arm, object, loss goal
                refine_objs[o] = o
                cup_id = self.s2r[o]
                observe = Observe()
                pose_dict = observe.control([cup_id])

                # Cup
                cup_pose = pose_dict[cup_id]
                pb.resetBasePositionAndOrientation(o, [cup_pose[0][0], cup_pose[0][1], 0.22], cup_pose[1])
                # Hook
                # hook_id = '1'
                # hook_pose = pose_dict[hook_id]
                # hook_pose = ((1,1,1),hook_pose[1])
                # pb.resetBasePositionAndOrientation(self.scn.hook, hook_pose[0], hook_pose[1])

                # obstacle = list(set(problem.fixed).difference(problem.marks))
                obstacles += problem.fixed

                pb.setRealTimeSimulation(1)
                time.sleep(1)
                pb.setRealTimeSimulation(0)
                reset.control()

            if name == 'pick':
                reset.control()
                a, b, p, g, c = args
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

                        # obstacles = problem.fixed #list(set(problem.fixed).difference(problem.marks))

                        
                        obstacles.append(refine_objs[b])
                
                        # print(f'obstcles = {obstacles}')
                        # print(f'refine_objs[b] = {refine_objs[b]}')
                        # print(f'fixed = {problem.fixed}')   
                        # input('wait')

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
                        action = ('pick', (a, refine_objs[b], p_new, g_new, c_new))
                    del refine_objs[b]

            commands = self.action2commands(problem, action)
            state = self.apply_commands(commands)



#######################################################

def main(verbose=True):
    rospy.init_node('executor', anonymous=True)
    # reset = Reset()
    # reset.apply()
    open_gripper()

if __name__ == '__main__':
    main()