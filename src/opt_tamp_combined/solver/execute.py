#!/usr/bin/env python

from __future__ import print_function
import os, sys
import pybullet as pb
import pickle

from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, get_max_limit, get_min_limit, LockRenderer
from examples.pybullet.utils.pybullet_tools.panda_primitives import Pose, get_grasp_gen, Attach, Detach, get_gripper_joints, GripperCommand, replanner


# from policies_edgepush import EdgePushPolicy
# from policies_retrieve import RetrievePolicy
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../')

from opt_tamp_retrieve.solver.policies import RetrievePolicy
from opt_tamp_edgepush.solver.policies import EdgePushPolicy




#######################################################

class Executor():
    def __init__(self, scenario):
        self.edgepush = EdgePushPolicy(scenario)
        self.retrieve = RetrievePolicy(scenario)

    def action2commands(self, problem, action, teleport=False):
        if action is None:
            return None
        # commands = []
        (name, args) = action 

        print(f'\nname = {name}\n') 
    
        if name == 'pick':
            a, b, p, g, c = args
            gripper_joint = get_gripper_joints(problem.robot, a)[0]
            position = get_max_limit(problem.robot, gripper_joint)
            open_gripper = GripperCommand(problem.robot, a, position, teleport=teleport)
            [t] = c.commands
            close_gripper = GripperCommand(problem.robot, a, g.grasp_width, teleport=teleport)
            attach = Attach(problem.robot, a, g, b)
            commands = [open_gripper, t, close_gripper, t.reverse()]
        elif name == 'place':
            a, b, p, g, c = args
            [t] = c.commands
            gripper_joint = get_gripper_joints(problem.robot, a)[0]
            position = get_max_limit(problem.robot, gripper_joint)
            open_gripper = GripperCommand(problem.robot, a, position, teleport=teleport)
            detach = Detach(problem.robot, a, b)
            commands = [t, open_gripper, t.reverse()]
        elif name == 'push':
            a, b, p, lg = args
            edgepush = self.edgepush 
            edgepush.specify(b, lg)
            commands = [edgepush]
        elif name == 'retrieve':
            a, b, p, lg = args
            bar = problem.tools[0]
            self.retrieve.specify(b, lg, bar)
            commands = [self.retrieve]    
        elif name == 'observe' or name == 'observe_retrieve' or name == 'observe_push':
            a, o, lg = args
            commands = []

        else:
            raise ValueError(name)
        print(name, args, commands) 
        return commands


    #######################################################

    def apply_command(self, commands):
        for i, command in enumerate(commands):
            print(command)
            pb.setRealTimeSimulation(1)
            command.control()


    #######################################################

    def execute(self, problem, plan):
        if plan is None:
            return None
        
        with open('./my_class_instance.pkl', 'wb') as file:
            pickle.dump(plan, file)
        
        observes = ['observe', 'observe_retrieve', 'observe_push']

        refine_objs = []

        for i, action in enumerate(plan):
            (name, args) = action
            if name in observes:
                a, o, lg = args
                refine_objs.append(o)

            if name == 'pick':
                a, b, p, g, c = args
                if b in refine_objs:
                    print(' Refine pick.')
                    saver = WorldSaver()

                    with LockRenderer(lock=False): 
                        
                        # Replan Grasp
                        grasp_fn = get_grasp_gen(problem, collisions=True)
                        g_list = grasp_fn(b)
                        
                        print(f'\ng_list = {g_list}\n')

                        saver.restore()
                        # Replan Trajectory

                        pb.setRealTimeSimulation(0)
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
                    action = ('pick', (a, b, p_new, g_new, c_new))
                    refine_objs.remove(b)

            commands = self.action2commands(problem, action)
            self.apply_command(commands)


#######################################################


#######################################################

def main(verbose=True):
    return 0

if __name__ == '__main__':
    main()