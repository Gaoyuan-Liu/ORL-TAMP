#!/usr/bin/env python


import sys, os, time
import pybullet as p
import moveit_commander
import cv2 as cv
import random
import itertools
import numpy as np


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')
sys.path.insert(0, file_path)



##########################
#         Import         #
##########################

# --------------------
# From current folder
# --------------------
from streams import get_cfree_approach_pose_test, get_cfree_pose_pose_test, get_cfree_traj_pose_test, get_near_pose_test,\
    get_cfree_traj_grasp_pose_test, BASE_CONSTANT, distance_fn, move_cost_fn
# from post_process import post_process, apply_commands, adjust_plan, execute
from neuro_generators import get_loose_goal_gen, get_retrieve_initial_test


# --------------------
# From pybullet tools
# --------------------
from examples.pybullet.utils.pybullet_tools.panda_primitives import Pose, Conf, get_ik_ir_gen, get_motion_gen, \
    get_stable_gen, get_grasp_gen, control_commands, get_arm_gen, State # apply_commands
from examples.pybullet.utils.pybullet_tools.panda_utils import get_arm_joints, ARM_NAMES, get_group_joints, open_arm, \
    get_group_conf, set_arm_conf
from examples.pybullet.utils.pybullet_tools.utils import connect, get_pose, get_euler, set_euler, is_placement, disconnect, pairwise_collision,\
    get_joint_positions, HideOutput, LockRenderer, wait_for_user, set_joint_positions, get_movable_joints, WorldSaver, has_gui, str_from_object, set_pose,\
    create_box, set_point, Point, TAN
from examples.pybullet.utils.pybullet_tools.panda_problems import create_panda, create_table, Problem

# -----------------------
# From pddlstream folder
# -----------------------
from pddlstream.algorithms.meta import create_parser, solve
from pddlstream.algorithms.common import SOLUTIONS
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, from_test
from pddlstream.language.constants import Equal, And, Or, print_solution, Exists, get_args, is_parameter, \
    get_parameter_name, PDDLProblem
from pddlstream.utils import read, INF, get_file_path, Profiler
from pddlstream.language.function import FunctionInfo
from pddlstream.language.stream import StreamInfo, DEBUG





class Solver():
    def __init__(self, robot, movable=[], tools=[], surfaces=[0], obstacles=[], marks=[]):
        self.algorithm = 'adaptive'
        self.verbose = False
        self.stream_info = {
        'inverse-kinematics': StreamInfo(),
        # 'test-cfree-pose-pose': StreamInfo(p_success=1e-3, verbose=self.verbose),
        'test-cfree-approach-pose': StreamInfo(p_success=1e-2, verbose=self.verbose),
        # 'test-cfree-traj-pose': StreamInfo(p_success=1e-1, verbose=self.verbose), # TODO: apply to arm and base trajs
        }
        self.render = True
        self.optimal = False
        self.max_time = 100
        self.initial_complexity = 1 #3
        self.max_complexity = INF
        self.success_cost = 0 if self.optimal else INF
        self.planner = 'ff-astar' if self.optimal else 'ff-wastar3'
        self.search_sample_ratio = 2 #2
        self.max_planner_time = 10 #5
        # effort_weight = 0 if args.optimal else 1
        self.effort_weight = 1e-3 if self.optimal else 1

        self.robot = robot
        self.problem = Problem(robot=robot)#, only_visual = [3])
        self.fixed_surfaces = surfaces
        self.obstacles = obstacles
        self.grasp_types = ('side', )
        self.movable = movable
        self.tools = tools
        self.marks = marks
        
        
        



    # --------------
    # Define Problem
    # --------------
    def problem_formation(self):
        self.problem.robot = self.robot
        self.problem.movable = self.movable 
        self.problem.grasp_types = self.grasp_types
        self.problem.surfaces = self.fixed_surfaces 
        self.problem.fixed += self.obstacles
        self.problem.tools = self.tools
        self.problem.marks = self.marks
        self.problem.side_grasps = self.movable
        

    # ------------------
    # Define PDDLProblem
    # ------------------
    def pddlstream_from_problem(self, collisions=True, teleport=False):

        domain_pddl = read(get_file_path(__file__, 'domain.pddl')) # It's a string
        stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
        constant_map = {}


        # Init
        init = [('Arm', self.robot), ('Controllable', self.robot), ('HandEmpty', self.robot)]
        for body in self.problem.movable:
            pose = Pose(body, get_pose(body), init=True) # TODO: supported here
            init += [('Graspable', body), ('Pose', body, pose),
                    ('AtPose', body, pose), ('Stackable', body, None)]
            
            # if pose.value[0][0] > 0.7:
            #     init += [('FarPose', body, pose)]

            for surface in self.problem.surfaces:
                # if is_placement(body, surface):
                # init += [('Supported', body, pose, surface)]
                init += [('Stackable', body, surface)]
        for body, ty in self.problem.body_types:
            init += [('Type', body, ty)]

        goal_literals = []
     
      

        # Goal
        # problem.goal -> goal_literal -> goal_formula
        # goal_literals = [tuple(['Holding', self.problem.goal_holding[0], self.problem.goal_holding[1]])]
        goal_literals = [self.goal]

        goal_formula = []
        for literal in goal_literals:
            parameters = [a for a in get_args(literal) if is_parameter(a)]
            if parameters: # This part is not using
                type_literals = [('Type', p, get_parameter_name(p)) for p in parameters]
                goal_formula.append(Exists(parameters, And(literal, *type_literals)))
            else:
                goal_formula.append(literal)
        goal_formula = And(*goal_formula) # This means total picking

        custom_limits = {}

        stream_map = {

            # Genrators
            # from_gen_fn basically just put the samples into a list.
            'sample-pose': from_gen_fn(get_stable_gen(self.problem, collisions=collisions)), 
            'sample-grasp': from_list_fn(get_grasp_gen(self.problem, collisions=collisions)),
            'inverse-kinematics': from_gen_fn(get_ik_ir_gen(self.problem, custom_limits=custom_limits,
                                                            collisions=collisions, teleport=teleport)),
            'sample-loose-goal': from_gen_fn(get_loose_goal_gen(self.problem, collisions=collisions)),


            # Testers
            'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(collisions=collisions)),
            'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(self.problem, collisions=collisions)),
            'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(self.robot, collisions=collisions)),
            'test-retrieve-initial-pose': from_test(get_retrieve_initial_test(self.problem, collisions=collisions)),
            'test-near-pose': from_test(get_near_pose_test(self.problem, collisions=collisions)),
        }
        return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal_formula)


    def solve(self, goal, collisions= True):

        self.goal = goal

        with HideOutput():
            self.problem_formation()
    
        saver = WorldSaver()
        pddlstream_problem = self.pddlstream_from_problem(collisions=collisions)

        # Print info
        _, _, _, stream_map, init, goal = pddlstream_problem
        print('Init:', init)
        print('Goal:', goal)
        print('Streams:', str_from_object(set(stream_map)))

        with Profiler(field='tottime', num=25): # cumtime | tottime
            with LockRenderer(lock=not self.render): # Lock the render
                solution = solve(pddlstream_problem, algorithm=self.algorithm, stream_info=self.stream_info,
                                planner=self.planner, max_planner_time=self.max_planner_time,
                                success_cost=self.success_cost, initial_complexity = self.initial_complexity, max_complexity = self.max_complexity,
                                max_time=self.max_time, verbose=True, debug=False, visualize=True,
                                unit_efforts=True, effort_weight=self.effort_weight,
                                search_sample_ratio=self.search_sample_ratio)
                saver.restore()
                
        # Put the ee tool aside
        gripper = self.problem.get_gripper()
        set_pose(gripper, ((10, 10, 0.2), (0,0,0,1)))
        return solution

    

    # -------------------
    # Dirty Work
    # -------------------
    def flip_checking(self, obj_poses):
        for i in obj_poses:
            eulers = get_euler(i)
            new_eulers = list(eulers)
            if abs(eulers[0]) >= 0.5:
                new_eulers = [0.0, eulers[1], eulers[2]]
                
            if abs(eulers[1]) >= 0.5:
                new_eulers = [eulers[0], 0.0, eulers[2]]
            set_euler(i, new_eulers)

    def get_bodies_from_type(self, problem):
        bodies_from_type = {}
        for body, ty in problem.body_types:
            bodies_from_type.setdefault(ty, set()).add(body)
        return bodies_from_type
            


  

if __name__ == '__main__':
    pass
   



