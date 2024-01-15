#!/usr/bin/env python

from __future__ import print_function

# from examples.pybullet.tamp.streams import get_cfree_approach_pose_test, get_cfree_pose_pose_test, get_cfree_traj_pose_test, \
#     get_cfree_traj_grasp_pose_test, BASE_CONSTANT, distance_fn, move_cost_fn

from examples.pybullet.utils.pybullet_tools.utils import connect, get_pose, is_placement, disconnect, \
    get_joint_positions, HideOutput, LockRenderer, wait_for_user
# from examples.pybullet.namo.stream import get_custom_limits

# from pddlstream.algorithms.meta import create_parser, solve
# from pddlstream.algorithms.common import SOLUTIONS
# from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, from_test
# from pddlstream.language.constants import Equal, And, print_solution, Exists, get_args, is_parameter, \
#     get_parameter_name, PDDLProblem
# from pddlstream.utils import read, INF, get_file_path, Profiler
# from pddlstream.language.function import FunctionInfo
# from pddlstream.language.stream import StreamInfo, DEBUG


from examples.pybullet.utils.pybullet_tools.utils import draw_base_limits, WorldSaver, has_gui, str_from_object, wait_for_duration

# from examples.pybullet.tamp.problems import PROBLEMS

from examples.pybullet.utils.pybullet_tools.panda_primitives import Pose, Conf, get_ik_ir_gen, get_motion_gen, get_ik_fn,\
    get_stable_gen, get_grasp_gen, Attach, Detach, Clean, Cook, control_commands, \
    get_gripper_joints, GripperCommand, apply_commands, State, replanner

from examples.pybullet.utils.pybullet_tools.utils import connect, get_pose, is_placement, point_from_pose, \
    disconnect, get_joint_positions, enable_gravity, save_state, restore_state, HideOutput, \
    get_distance, LockRenderer, get_min_limit, get_max_limit, has_gui, WorldSaver, wait_if_gui, add_line, SEPARATOR

import pybullet as pb



from policies_sb3 import HookingPolicy, HookEnvNominal, Observe


    
class Executor():
    def __init__(self, problem, scenario):
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
            hook = HookingPolicy(scenario)
            hook.specify(b, lg)
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
        return state        


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