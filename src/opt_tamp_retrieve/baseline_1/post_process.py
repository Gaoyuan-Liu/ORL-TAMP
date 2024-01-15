#!/usr/bin/env python

from __future__ import print_function

import os, sys

from examples.pybullet.utils.pybullet_tools.utils import LockRenderer, WorldSaver, wait_for_duration, get_max_limit, wait_if_gui

from examples.pybullet.utils.pybullet_tools.panda_primitives import Pose, get_grasp_gen, Attach, Detach, \
     get_gripper_joints, GripperCommand, State, replanner

import pybullet as pb


from policies import HookPolicyIdeal, Observe

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
from motion_planner import CartesianPlanner


    
class Executor():
    def __init__(self):
        self.cartesian_planner = CartesianPlanner()
        pass

#######################################################


    def action2commands(self, problem, action, scenario, teleport=False):
        if action is None:
            return None
        # commands = []
        (name, args) = action 

        print(f'\nname = {name}\n') 
    
        if name == 'pick':
            a, b, p, g, _, c = args
            [t] = c.commands
            close_gripper = GripperCommand(problem.robot, a, g.grasp_width, teleport=teleport)
            attach = Attach(problem.robot, a, g, b)
            commands = [t, close_gripper, attach, t.reverse()]
        elif name == 'place':
            a, b, p, g, _, c = args
            [t] = c.commands
            gripper_joint = get_gripper_joints(problem.robot, a)[0]
            position = get_max_limit(problem.robot, gripper_joint)
            open_gripper = GripperCommand(problem.robot, a, position, teleport=teleport)
            detach = Detach(problem.robot, a, b)
            commands = [t, detach, open_gripper, t.reverse()]
        elif name == 'hook':
            a, b, p, lg = args
            hook = HookPolicyIdeal(scenario, b, p, lg, self.cartesian_planner)
            # hook.specify(b, lg)
            # print(f'\ngoal = {lg.value}\n')
            commands = [hook]
        elif name == 'observe':
            a, o, lg = args
            observe = Observe()
            commands = [observe]

        else:
            raise ValueError(name)
        print(name, args, commands) 
        return commands




    #######################################################

    def apply_commands(self, state, commands, time_step=None, pause=False, **kwargs):
        for i, command in enumerate(commands):
            print(i, command)
            if command.type == 'hook':
                command.apply(state, **kwargs)  
                # print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh\n')
            elif command.type == 'observe':
                print('observe')
            else:
                # print(f'else command = {command.type}')
                for j, _ in enumerate(command.apply(state, **kwargs)):
                    state.assign()
                    if j == 0:
                        continue
                    if time_step is None:
                        wait_for_duration(1e-2)
                        wait_if_gui('Command {}, Step {}) Next?'.format(i, j))
                    else:
                        wait_for_duration(time_step)
            if pause:
                wait_if_gui()


    #######################################################

    def execute(self, problem, plan, scenario):
        if plan is None:
            return None
        state = State()
        for i, action in enumerate(plan):
            (name, args) = action
            if name == 'observe':
                next_name, next_args = plan[i+1]
                if next_name == 'pick':

                    a, b, p, g, _, c = next_args
                    saver = WorldSaver()
                    with LockRenderer(lock=True): 
                        
                        # Replan Grasp
                        grasp_fn = get_grasp_gen(problem, collisions=True)
                        g_list = grasp_fn(b)
                        saver.restore()
                        
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
                        saver.restore()

                    plan[i+1] = (next_name, (a, b, p_new, g_new, _, c_new))


            commands = self.action2commands(problem, action, scenario)
            state = self.apply_commands(state, commands, time_step=0.01)



#######################################################

def main(verbose=True):
    return 0

if __name__ == '__main__':
    main()