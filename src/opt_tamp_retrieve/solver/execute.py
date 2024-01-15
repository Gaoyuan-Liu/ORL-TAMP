#!/usr/bin/env python

from __future__ import print_function

from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, LockRenderer, get_max_limit

from examples.pybullet.utils.pybullet_tools.panda_primitives import Pose, get_grasp_gen, Attach, Detach, \
    get_gripper_joints, GripperCommand, State, replanner


import pybullet as pb



from policies import RetrievePolicy


class Observe():
    def __init__(self):
        self.type = 'observe'

    def control(self):
        pass

#######################################################
    
class Executor():
    def __init__(self, scn):
        self.retrieve = RetrievePolicy(scn)


    def action2commands(self, problem, action, teleport=False):
        if action is None:
            return None

        (name, args) = action 

        print(f'\nname = {name}\n') 
    
        if name == 'pick':
            a, b, p, g, _, c = args
            gripper_joint = get_gripper_joints(problem.robot, a)[0]
            position = get_max_limit(problem.robot, gripper_joint)
            open_gripper = GripperCommand(problem.robot, a, position, teleport=teleport)
            [t] = c.commands
            close_gripper = GripperCommand(problem.robot, a, g.grasp_width, teleport=teleport)
            attach = Attach(problem.robot, a, g, b)
            commands = [open_gripper, t, close_gripper, t.reverse()]

        elif name == 'place':
            a, b, p, g, _, c = args
            [t] = c.commands
            gripper_joint = get_gripper_joints(problem.robot, a)[0]
            position = get_max_limit(problem.robot, gripper_joint)
            open_gripper = GripperCommand(problem.robot, a, position, teleport=teleport)
            detach = Detach(problem.robot, a, b)
            commands = [t, open_gripper, t.reverse()]

        elif name == 'retrieve':
            a, b, p, lg = args
            bar = problem.tools[0]
            self.retrieve.specify(b, lg, bar)
            commands = [self.retrieve]

        elif name == 'observe':
            a, b, lg = args
            observe = Observe()
            commands = [observe]

        else:
            raise ValueError(name)
        print(name, args, commands) 
        return commands




    #######################################################

    def apply_commands(self, commands):
        pb.setRealTimeSimulation(1)
        for i, command in enumerate(commands):
            command.control()
       


    #######################################################

    def execute(self, problem, plan):
        if plan is None:
            return None
        state = State()

        refine_objs = []
        for i, action in enumerate(plan):
            (name, args) = action
            if name == 'observe':
                # Get the object
                a, o, lg = args # Arm, object, loss goal
                refine_objs.append(o)

            if name == 'pick':
                print("\033[32m {}\033[00m" .format(' Plan Refine'))
                a, b, p, g, _, c = args
                if b in refine_objs:
                    saver = WorldSaver()
                    with LockRenderer(lock=True): 
                        
                        # Replan Grasp
                        grasp_fn = get_grasp_gen(problem, collisions=True)
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

                        action = ('pick', (a, b, p_new, g_new, _, c_new))
                        refine_objs.remove(b)

            commands = self.action2commands(problem, action)
            self.apply_commands(commands)



#######################################################

def main(verbose=True):
    return 0

if __name__ == '__main__':
    main()