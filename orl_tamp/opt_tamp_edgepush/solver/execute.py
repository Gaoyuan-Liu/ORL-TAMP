#!/usr/bin/env python

from __future__ import print_function
import os, sys, time
# from examples.pybullet.tamp.streams import get_cfree_approach_pose_test, get_cfree_pose_pose_test, get_cfree_traj_pose_test, \
#     get_cfree_traj_grasp_pose_test, BASE_CONSTANT, distance_fn, move_cost_fn

from examples.pybullet.utils.pybullet_tools.utils import connect, get_pose, is_placement, disconnect, \
    get_joint_positions, HideOutput, LockRenderer, wait_for_user



from examples.pybullet.utils.pybullet_tools.utils import draw_base_limits, WorldSaver, has_gui, str_from_object, wait_for_duration



from examples.pybullet.utils.pybullet_tools.panda_primitives import Pose, Conf, get_ik_ir_gen, get_motion_gen, get_ik_fn,\
    get_stable_gen, get_grasp_gen, Attach, Detach, Clean, Cook, control_commands, \
    get_gripper_joints, GripperCommand, apply_commands, State, replanner

from examples.pybullet.utils.pybullet_tools.utils import connect, get_pose, is_placement, point_from_pose, \
    disconnect, get_joint_positions, enable_gravity, save_state, restore_state, HideOutput, \
    get_distance, LockRenderer, get_min_limit, get_max_limit, has_gui, WorldSaver, wait_if_gui, add_line, SEPARATOR

import pybullet as pb




from policies import EdgePushPolicy, Observe
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')



class Observe():
    def __init__(self):
        self.type = 'observe'

    def control(self):
        pass
    


#######################################################

class Execute():
    def __init__(self, scenario):
        self.edgepush = EdgePushPolicy(scenario)

    def action2commands(self, problem, action):
        if action is None:
            return None
        # commands = []
        (name, args) = action 

        print(f'\nname = {name}\n') 
    
        if name == 'pick':
            a, b, p, g, _, c = args
            gripper_joint = get_gripper_joints(problem.robot, a)[0]
            position = get_max_limit(problem.robot, gripper_joint)
            open_gripper = GripperCommand(problem.robot, a, position)
            [t] = c.commands
            close_gripper = GripperCommand(problem.robot, a, g.grasp_width)
            attach = Attach(problem.robot, a, g, b)
            # commands = [t, close_gripper, attach, t.reverse()]
            commands = [open_gripper, t, close_gripper, t.reverse()]
        elif name == 'place':
            a, b, p, g, _, c = args
            [t] = c.commands
            gripper_joint = get_gripper_joints(problem.robot, a)[0]
            position = get_max_limit(problem.robot, gripper_joint)
            open_gripper = GripperCommand(problem.robot, a, position)
            detach = Detach(problem.robot, a, b)
            # commands = [t, detach, open_gripper, t.reverse()]
            commands = [t, open_gripper, t.reverse()]
        elif name == 'push':
            a, b, p, lg = args
            edgepush = self.edgepush # EdgePushPolicy(scenario)
            edgepush.specify(b, lg)
            gripper_joint = get_gripper_joints(problem.robot, a)[0]
            position = get_max_limit(problem.robot, gripper_joint)
            open_gripper = GripperCommand(problem.robot, a, position)
            commands = [edgepush]
        elif name == 'observe':
            a, o, lg = args
            observe = Observe()
            commands = [Observe()]

        else:
            raise ValueError(name)
        print(name, args, commands) 
        return commands


    #######################################################

    def apply_commands(self, commands, time_step=None, problem=None, **kwargs):
        pb.setRealTimeSimulation(1)
        for i, command in enumerate(commands):
            command.control()
            
        


    #######################################################

    def execute(self, problem, plan):
        if plan is None:
            return None
        for i, action in enumerate(plan):
            
            (name, args) = action
            if name == 'observe':
                next_name, next_args = plan[i+1]
                if next_name == 'pick':

                    a, b, p, g, _, c = next_args
                    saver = WorldSaver()
                    with LockRenderer(lock=False): 
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

            
            commands = self.action2commands(problem, action)
            self.apply_commands(commands, time_step=0.01, problem=problem)



#######################################################

def main(verbose=True):
    return 0

if __name__ == '__main__':
    main()