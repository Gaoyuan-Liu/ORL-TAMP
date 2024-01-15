from __future__ import print_function

import copy
import pybullet as pb
import random
import time, os, sys
from itertools import islice, count

import numpy as np

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path)

from .ikfast.franka_panda.ik import panda_inverse_kinematics
from .ikfast.utils import USE_CURRENT, USE_ALL
from .panda_problems import get_fixed_bodies
from .panda_utils import TOP_HOLDING_LEFT_ARM, SIDE_HOLDING_LEFT_ARM, GET_GRASPS, get_gripper_joints, \
    get_carry_conf, get_top_grasps, get_side_grasps, get_edge_grasps, open_arm, arm_conf, get_gripper_link, get_arm_joints, \
    learned_pose_generator, PANDA_TOOL_FRAMES, get_x_presses, PANDA_GROUPS, joints_from_names, \
    is_drake_pr2, get_group_joints, get_group_conf, compute_grasp_width, PANDA_GRIPPER_ROOTS, \
    get_top_cylinder_grasps, get_side_cylinder_grasps
from .utils import invert, multiply, get_name, set_pose, get_link_pose, is_placement, get_center_extent,\
    pairwise_collision, set_joint_positions, get_joint_positions, sample_placement, get_pose, waypoints_from_path, \
    unit_quat, plan_base_motion, plan_joint_motion, base_values_from_pose, pose_from_base_values, \
    uniform_pose_generator, sub_inverse_kinematics, add_fixed_constraint, remove_debug, remove_fixed_constraint, \
    disable_real_time, enable_gravity, joint_controller_hold, get_distance, \
    get_min_limit, user_input, step_simulation, get_body_name, get_bodies, BASE_LINK, \
    add_segments, get_max_limit, link_from_name, BodySaver, get_aabb, Attachment, interpolate_poses, \
    plan_direct_joint_motion, has_gui, create_attachment, wait_for_duration, get_extend_fn, set_renderer, \
    get_custom_limits, all_between, get_unit_vector, wait_if_gui, \
    set_base_values, euler_from_quat, INF, elapsed_time, get_moving_links, flatten_links, get_relative_pose, \
    get_movable_joints, get_sample_fn, inverse_kinematics, joint_controller, enable_real_time, refine_path, flatten, set_client, get_client

from .transformations import quaternion_from_euler
import math

BASE_EXTENT = 3.5 # 2.5
BASE_LIMITS = (-BASE_EXTENT*np.ones(2), BASE_EXTENT*np.ones(2))
GRASP_LENGTH = 0.02 #0.02
APPROACH_DISTANCE = 0.03 + GRASP_LENGTH 
SELF_COLLISIONS = True
FRANKA_CUSTOM_LIMITS= {0:(-2.8, 2.8), 1:(-1.7, 1.7), 2:(-2.8, 2.8), 3:(-3, 0), 4:(-2.8, 2.8),
                       5:(0, 3.7), 6:(-2.8, 2.8)}

##################################################

# def get_base_limits(robot):
#     if is_drake_pr2(robot):
#         joints = get_group_joints(robot, 'base')[:2]
#         lower = [get_min_limit(robot, j) for j in joints]
#         upper = [get_max_limit(robot, j) for j in joints]
#         return lower, upper
#     return BASE_LIMITS

##################################################

# For motion planning
class BodyConf(object):
    num = count()
    def __init__(self, body, configuration=None, joints=None):
        if joints is None:
            joints = get_movable_joints(body)
            # print(f'\njoints = {joints}\n')
        if configuration is None:
            configuration = get_joint_positions(body, joints)
        self.body = body
        self.joints = joints
        self.configuration = configuration
        self.index = next(self.num)
    @property
    def values(self):
        return self.configuration
    def assign(self):
        set_joint_positions(self.body, self.joints, self.configuration)
        return self.configuration
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'q{}'.format(index)
    

class BodyPose(object):
    num = count()
    def __init__(self, body, pose=None):
        if pose is None:
            pose = get_pose(body)
        self.body = body
        self.pose = pose
        self.index = next(self.num)
    @property
    def value(self):
        return self.pose
    def assign(self):
        set_pose(self.body, self.pose)
        return self.pose
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'p{}'.format(index)
    

def get_tool_link(robot):
    # return link_from_name(robot, TOOL_FRAMES[get_body_name(robot)])
    return link_from_name(robot, 'panda_hand')


def get_simple_ik_fn(robot, fixed=[], teleport=False, num_attempts=50):
    movable_joints = get_movable_joints(robot)
    # movable_joints = [0,1,2,3,4,5,6,9,10]
    # print(f'\nmovable_joints = {movable_joints}\n')
    sample_fn = get_sample_fn(robot, movable_joints, custom_limits=FRANKA_CUSTOM_LIMITS)
    tool_link = get_tool_link(robot)
    def fn(pose):
        obstacles = fixed
        # approach_pose = approach_from_grasp(grasp.approach_pose, gripper_pose)
        for _ in range(num_attempts):
            set_joint_positions(robot, movable_joints, sample_fn()) # Random seed
            # TODO: multiple attempts?
            q = inverse_kinematics(robot, tool_link, pose)
            if (q is None) or any(pairwise_collision(robot, b) for b in obstacles):
                continue
            conf = BodyConf(robot, q)
            return conf
            # TODO: holding collisions
        return None
    return fn

def assign_fluent_state(fluents):
    obstacles = []
    for fluent in fluents:
        name, args = fluent[0], fluent[1:]
        if name == 'atpose':
            o, p = args
            obstacles.append(o)
            p.assign()
        else:
            raise ValueError(name)
    return obstacles

class BodyPath(object):
    def __init__(self, body, path, joints=None, attachments=[]):
        if joints is None:
            joints = get_movable_joints(body)
        self.body = body
        self.path = path
        self.joints = joints
        self.attachments = attachments
    def bodies(self):
        return set([self.body] + [attachment.body for attachment in self.attachments])
    def iterator(self):
        # TODO: compute and cache these
        # TODO: compute bounding boxes as well
        for i, configuration in enumerate(self.path):
            set_joint_positions(self.body, self.joints, configuration)
            for grasp in self.attachments:
                grasp.assign()
            yield i
    def control(self, real_time=False, dt=0):
        # TODO: just waypoints
        if real_time:
            enable_real_time()
        else:
            disable_real_time()
        for values in self.path:
            for _ in joint_controller(self.body, self.joints, values):
                enable_gravity()
                if not real_time:
                    step_simulation()
                time.sleep(dt)
    # def full_path(self, q0=None):
    #     # TODO: could produce sequence of savers
    def refine(self, num_steps=0):
        return self.__class__(self.body, refine_path(self.body, self.joints, self.path, num_steps), self.joints, self.attachments)
    def reverse(self):
        return self.__class__(self.body, self.path[::-1], self.joints, self.attachments)
    def __repr__(self):
        return '{}({},{},{},{})'.format(self.__class__.__name__, self.body, len(self.joints), len(self.path), len(self.attachments))

DEBUG_FAILURE = False
def get_free_motion_gen(robot, fixed=[], teleport=False, self_collisions=True, algorithm=None):
    def fn(conf1, conf2, fluents=[]):
        assert ((conf1.body == conf2.body) and (conf1.joints == conf2.joints))
        if teleport:
            path = [conf1.configuration, conf2.configuration]
        else:
            conf1.assign()
            obstacles = fixed + assign_fluent_state(fluents)
            path = plan_joint_motion(robot, conf2.joints, conf2.configuration, obstacles=obstacles, self_collisions=self_collisions)#, algorithm=algorithm)
            if path is None:
                # print('\n path is None.\n')
                if DEBUG_FAILURE: wait_if_gui('Free motion failed')
                return None
        command = CommandMP([BodyPath(robot, path, joints=conf2.joints)])
        return (command,)
    return fn


class CommandMP(object):
    num = count()
    def __init__(self, body_paths):
        self.body_paths = body_paths
        self.index = next(self.num)
    def bodies(self):
        return set(flatten(path.bodies() for path in self.body_paths))
    # def full_path(self, q0=None):
    #     if q0 is None:
    #         q0 = Conf(self.tree)
    #     new_path = [q0]
    #     for partial_path in self.body_paths:
    #         new_path += partial_path.full_path(new_path[-1])[1:]
    #     return new_path
    def step(self):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                msg = '{},{}) step?'.format(i, j)
                wait_if_gui(msg)
                #print(msg)
    def execute(self, time_step=0.05):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                #time.sleep(time_step)
                wait_for_duration(time_step)
    def control(self, real_time=False, dt=0): # TODO: real_time
        for body_path in self.body_paths:
            body_path.control(real_time=real_time, dt=dt)
    def refine(self, **kwargs):
        return self.__class__([body_path.refine(**kwargs) for body_path in self.body_paths])
    def reverse(self):
        return self.__class__([body_path.reverse() for body_path in reversed(self.body_paths)])
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'c{}'.format(index)

##################################################

class Pose(object):
    num = count()
    #def __init__(self, position, orientation):
    #    self.position = position
    #    self.orientation = orientation
    def __init__(self, body, value=None, support=None, init=False):
        self.body = body
        if value is None:
            value = get_pose(self.body)
        self.value = tuple(value)
        self.support = support
        self.init = init
        self.index = next(self.num)
    @property
    def bodies(self):
        return flatten_links(self.body)
    def assign(self):
        set_pose(self.body, self.value)
    def iterate(self):
        yield self
    def to_base_conf(self):
        values = base_values_from_pose(self.value)
        return Conf(self.body, range(len(values)), values)
    def __repr__(self):
        index = self.index
        return 'p{}'.format(index)

class Grasp(object):
    def __init__(self, grasp_type, body, value, approach, carry):
        self.grasp_type = grasp_type
        self.body = body
        self.value = tuple(value) # gripper_from_object
        self.approach = tuple(approach)
        self.carry = tuple(carry)
    # def get_attachment(self, robot, arm):
    #     tool_link = link_from_name(robot, 'panda_hand') #link_from_name(robot, PANDA_TOOL_FRAMES[arm])
    #     return Attachment(robot, tool_link, self.value, self.body)
    def get_attachment(self, arm, arm_): 
        tool_link = link_from_name(arm, 'panda_hand') #link_from_name(robot, PANDA_TOOL_FRAMES[arm])
        return Attachment(arm, tool_link, self.value, self.body) #self.value
    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)

class Conf(object):
    def __init__(self, body, joints, values=None, init=False):
        self.body = body
        self.joints = joints
        if values is None:
            values = get_joint_positions(self.body, self.joints)
        self.values = tuple(values)
        self.init = init
    @property
    def bodies(self): # TODO: misnomer
        return flatten_links(self.body, get_moving_links(self.body, self.joints))
    def assign(self):
        set_joint_positions(self.body, self.joints, self.values)
    def iterate(self):
        yield self
    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)

#####################################

class Command(object):
    def control(self, dt=0):
        raise NotImplementedError()
    def apply(self, state, **kwargs):
        raise NotImplementedError()
    def iterate(self):
        raise NotImplementedError()

class Commands(object):
    def __init__(self, state, savers=[], commands=[]):
        self.state = state
        self.savers = tuple(savers)
        self.commands = tuple(commands)
    def assign(self):
        for saver in self.savers:
            saver.restore()
        return copy.copy(self.state)
    def apply(self, state, **kwargs):
        for command in self.commands:
            for result in command.apply(state, **kwargs):
                yield result
    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)

#####################################

class Trajectory(Command):
    _draw = False
    def __init__(self, path):
        self.path = tuple(path)
        self.type = 'Trajectory'
        # TODO: constructor that takes in this info
    def apply(self, state, sample=1):
        handles = add_segments(self.to_points()) if self._draw and has_gui() else []
        for conf in self.path[::sample]:
            conf.assign()
            yield
        end_conf = self.path[-1]
        if isinstance(end_conf, Pose):
            state.poses[end_conf.body] = end_conf
        for handle in handles:
            remove_debug(handle)
    def control(self, dt=0.1, **kwargs):
        # TODO: just waypoints
        for conf in self.path:
            if isinstance(conf, Pose):
                conf = conf.to_base_conf()
            # for _ in joint_controller_hold(conf.body, conf.joints, conf.values, max_force=1002):
            #     step_simulation()
            #     time.sleep(dt)
            pb.setJointMotorControlArray(conf.body, conf.joints, pb.POSITION_CONTROL, conf.values, forces=[100]*len(conf.joints))
            time.sleep(dt)
            # for i in range(10):
            #     pb.stepSimulation()

                
    def to_points(self, link=BASE_LINK):
        # TODO: this is computationally expensive
        points = []
        for conf in self.path:
            with BodySaver(conf.body):
                conf.assign()
                #point = np.array(point_from_pose(get_link_pose(conf.body, link)))
                point = np.array(get_group_conf(conf.body, 'base'))
                point[2] = 0
                point += 1e-2*np.array([0, 0, 1])
                if not (points and np.allclose(points[-1], point, atol=1e-3, rtol=0)):
                    points.append(point)
        points = get_target_path(self)
        return waypoints_from_path(points)
    def distance(self, distance_fn=get_distance):
        total = 0.
        for q1, q2 in zip(self.path, self.path[1:]):
            total += distance_fn(q1.values, q2.values)
        return total
    def iterate(self):
        for conf in self.path:
            yield conf
    def reverse(self):
        return Trajectory(reversed(self.path))
    #def __repr__(self):
    #    return 't{}'.format(id(self) % 1000)
    def __repr__(self):
        d = 0
        if self.path:
            conf = self.path[0]
            d = 3 if isinstance(conf, Pose) else len(conf.joints)
        return 't({},{})'.format(d, len(self.path))

def create_trajectory(robot, joints, path):
    return Trajectory(Conf(robot, joints, q) for q in path)

##################################################

class GripperCommand(Command):
    def __init__(self, robot, arm, position, teleport=False):
        self.robot = robot
        self.arm = arm
        self.position = position
        self.teleport = teleport
        self.type = 'GripperCommand'
    def apply(self, state, **kwargs):
        joints = get_gripper_joints(self.robot, self.arm)
        start_conf = get_joint_positions(self.robot, joints)
        end_conf = [self.position] * len(joints)
        if self.teleport:
            path = [start_conf, end_conf]
        else:
            extend_fn = get_extend_fn(self.robot, joints)
            path = [start_conf] + list(extend_fn(start_conf, end_conf))
        for positions in path:
            set_joint_positions(self.robot, joints, positions)
            yield positions
    def control(self, **kwargs):
        joints = get_gripper_joints(self.robot, self.arm)
        positions = [self.position]*len(joints)
        pb.setJointMotorControlArray(self.robot, joints, pb.POSITION_CONTROL, positions, forces=[10, 10])
        time.sleep(0.5)
        # for _ in joint_controller_hold(self.robot, joints, positions):
        #     yield

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, self.position)
    
class Attach(Command):
    vacuum = True
    def __init__(self, robot, arm, grasp, body):
        self.robot = robot
        self.arm = arm
        self.grasp = grasp
        self.body = body
        self.link = link_from_name(self.robot, 'panda_hand') #link_from_name(self.robot, PANDA_TOOL_FRAMES.get(self.arm, self.arm))
        self.type = 'Attach'
        #self.attachment = None
    def assign(self):
        gripper_pose = get_link_pose(self.robot, self.link)
        body_pose = multiply(gripper_pose, self.grasp.value)
        set_pose(self.body, body_pose)
    def apply(self, state, **kwargs):
        state.attachments[self.body] = create_attachment(self.robot, self.link, self.body)
        state.grasps[self.body] = self.grasp
        del state.poses[self.body]
        yield
    def control(self, dt=0, **kwargs):
        if self.vacuum:
            add_fixed_constraint(self.body, self.robot, self.link)
            #add_fixed_constraint(self.body, self.robot, self.link, max_force=1) # Less force makes it easier to pick
        else:
            # TODO: the gripper doesn't quite work yet
            gripper_name = 'gripper'
            joints = joints_from_names(self.robot, PANDA_GROUPS[gripper_name])
            values = [get_min_limit(self.robot, joint) for joint in joints] # Closed
            for _ in joint_controller_hold(self.robot, joints, values):
                step_simulation()
                time.sleep(dt)
    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))

class Detach(Command):
    def __init__(self, robot, arm, body):
        self.robot = robot
        self.arm = arm
        self.body = body
        self.link = link_from_name(self.robot, 'panda_hand') #link_from_name(self.robot, PANDA_TOOL_FRAMES.get(self.arm, self.arm))
        self.type = "Detach"
        # TODO: pose argument to maintain same object
    def apply(self, state, **kwargs):
        del state.attachments[self.body]
        state.poses[self.body] = Pose(self.body, get_pose(self.body))
        del state.grasps[self.body]
        yield
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)
    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))

##################################################

class Clean(Command):
    def __init__(self, body):
        self.body = body
    def apply(self, state, **kwargs):
        state.cleaned.add(self.body)
        self.control()
        yield
    def control(self, **kwargs):
        p.addUserDebugText('Cleaned', textPosition=(0, 0, .25), textColorRGB=(0,0,1), #textSize=1,
                           lifeTime=0, parentObjectUniqueId=self.body)
        #p.setDebugObjectColor(self.body, 0, objectDebugColorRGB=(0,0,1))
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

class Cook(Command):
    # TODO: global state here?
    def __init__(self, body):
        self.body = body
    def apply(self, state, **kwargs):
        state.cleaned.remove(self.body)
        state.cooked.add(self.body)
        self.control()
        yield
    def control(self, **kwargs):
        # changeVisualShape
        # setDebugObjectColor
        #p.removeUserDebugItem # TODO: remove cleaned
        p.addUserDebugText('Cooked', textPosition=(0, 0, .5), textColorRGB=(1,0,0), #textSize=1,
                           lifeTime=0, parentObjectUniqueId=self.body)
        #p.setDebugObjectColor(self.body, 0, objectDebugColorRGB=(1,0,0))
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

##################################################

# class Push(Command):
#     def __init__(self, push_t):
#         self.t = push_t
#     def apply(self, state, **kwargs):
#         control_joints(robot, arm_joints, point.positions)
#         control_joints(robot, finger_joints, [0.0,0.0])
#         time.sleep(0.01)
        
    
#     def __repr__(self):
#         return 'pu{}'.format(id(self) % 1000)



##################################################
def get_arm_gen(problem, collisions=False, randomize=True):
    def fn(robots, body, pose):
        # robots = problem.robots
        # return [(a,) for a in robots]
        while True:
            if pose.value[0][1] > 0.7:
                a = robots[1]
            # elif pose.value[0][1] < 0.3:
            #     a = robots[0]
            else:
                a = robots[0]
            # a = random.choice(robots)
            yield (a,)
    return fn

##################################################

def get_grasp_gen(problem, collisions=False, randomize=False):
    for grasp_type in problem.grasp_types:
        if grasp_type not in GET_GRASPS:
            raise ValueError('Unexpected grasp type:', grasp_type)
    def fn(body):
        # TODO: max_grasps
        # TODO: return grasps one by one
        grasps = []
        type = 'panda'
        # if 'top' in problem.grasp_types:
        #     approach_vector = APPROACH_DISTANCE*get_unit_vector([0, 0, 1])
        #     grasps.extend(Grasp('top', body, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_LEFT_ARM)
        #                   for g in get_top_grasps(body, grasp_length=GRASP_LENGTH))
        # if 'side' in problem.grasp_types:
        #     approach_vector = APPROACH_DISTANCE*get_unit_vector([0, 0, 1])
        #     grasps.extend(Grasp('side', body, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_LEFT_ARM)
        #                   for g in get_side_grasps(body, grasp_length=GRASP_LENGTH))
        # if 'edge' in problem.grasp_types:
        #     approach_vector = APPROACH_DISTANCE*get_unit_vector([0, 0, 1])
        #     grasps.extend(Grasp('edge', body, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_LEFT_ARM)
        #                   for g in get_edge_grasps(body, grasp_length=GRASP_LENGTH))    
        
        
        if body in problem.top_grasps:
            approach_vector = APPROACH_DISTANCE*get_unit_vector([0, 0, 1])
            grasps.extend(Grasp('top', body, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_LEFT_ARM)
                          for g in get_top_grasps(body, grasp_length=GRASP_LENGTH))
        if body in problem.edge_grasps:
            approach_vector = APPROACH_DISTANCE*get_unit_vector([0, 0, 1])
            grasps.extend(Grasp('edge', body, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_LEFT_ARM)
                          for g in get_edge_grasps(body, grasp_length=GRASP_LENGTH))    
            
        if body in problem.side_grasps:
            approach_vector = APPROACH_DISTANCE*get_unit_vector([0, 0, 1])
            grasps.extend(Grasp('side', body, g, multiply((approach_vector, unit_quat()), g), SIDE_HOLDING_LEFT_ARM)
                          for g in get_side_grasps(body, grasp_length=GRASP_LENGTH))

        
        filtered_grasps = []
        for grasp in grasps:
            grasp_width = compute_grasp_width(problem.robot, type, body, grasp.value) if collisions else 0.0
            if grasp_width is not None:
                grasp.grasp_width = grasp_width
                filtered_grasps.append(grasp)
        # if randomize:
        #     random.shuffle(filtered_grasps)
        return [(g,) for g in filtered_grasps]
        #for g in filtered_grasps:
        #    yield (g,)
    return fn

##################################################

##################################################

def accelerate_gen_fn(gen_fn, max_attempts=1):
    def new_gen_fn(*inputs):
        generator = gen_fn(*inputs)
        while True:
            for i in range(max_attempts):
                try:
                    output = next(generator)
                except StopIteration:
                    return
                if output is not None:
                    print(gen_fn.__name__, i)
                    yield output
                    break
    return new_gen_fn


##################################################


def get_stable_gen(problem, collisions=True, **kwargs):
    # obstacles = list(set(problem.fixed).difference(problem.marks)) if collisions else []
    def gen(body, surface):
        # TODO: surface poses are being sampled in pr2_belief
        if surface is None:
            surfaces = problem.surfaces
        else:
            surfaces = [surface]
        # surfaces = problem.surfaces
        i = 0
        while True:
            surface = random.choice(surfaces) # TODO: weight by area
            
            # if body in problem.tools:
            #     body_pose = get_pose(surface)
            #     center, extent = get_center_extent(body)
            #     body_pose = ((body_pose[0][0], body_pose[0][1], extent[2]/2), body_pose[1]) # pos and orn
            # elif 'edge' in problem.grasp_types:
            #     body_pose = get_pose(surface)
            #     center, extent = get_center_extent(body)
            #     random_angle = random.uniform(-math.pi, math.pi)
            #     # random_angle = random.uniform(0, math.pi)
            #     quat = quaternion_from_euler(0, 0, random_angle, axes='sxyz')
            #     body_pose = ((0.5, -0.26, 0.62), quat)
            # else:
            #     body_pose = sample_placement(body, surface, **kwargs)

            ################

            if body in problem.edge_grasps:
                pose = pb.getBasePositionAndOrientation(body)
                body_pose = get_pose(surface)
                center, extent = get_center_extent(body)

                random_angle = random.uniform(-math.pi, math.pi)
                quat = quaternion_from_euler(0, 0, random_angle, axes='sxyz')

                random_x = random.uniform(0.4, 0.6)
                # body_pose = ((0.5, -0.52, 0.34), quat)
                body_pose = ((random_x, -0.35, 0.47), quat)
                if body_pose is None:
                    break
                p = Pose(body, body_pose, surface)
                p.assign()
                yield (p,)

            else:
                body_pose = sample_placement(body, surface, **kwargs)


            ################
            if body_pose is None:
                break
            p = Pose(body, body_pose, surface)
            p.assign()
            # time.sleep(1)
            # if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
            yield (p,)
    # TODO: apply the acceleration technique here
    return gen


##################################################

def get_pose_gen(problem, collisions=True, **kwargs):
    # obstacles = problem.fixed if collisions else []
    def gen(body, surface):
        # TODO: surface poses are being sampled in pr2_belief
        if surface is None:
            surfaces = problem.surfaces
        else:
            surfaces = [surface]
        # surfaces = problem.surfaces
        while True:
            surface = random.choice(surfaces) # TODO: weight by area
            body_pose = get_pose(surface)#sample_placement(body, surface, **kwargs)
            if body_pose is None:
                break
            p = Pose(body, body_pose, surface)
            p.assign()
            # if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
            yield (p,)
    # TODO: apply the acceleration technique here
    return gen


##################################################

def get_tool_from_root(robot, arm):
    root_link = link_from_name(robot, PANDA_GRIPPER_ROOTS[arm])
    tool_link = link_from_name(robot, PANDA_TOOL_FRAMES[arm])
    return get_relative_pose(robot, root_link, tool_link)

def iterate_approach_path(robot, arm, gripper, pose, grasp, body=None):
    tool_from_root = get_tool_from_root(robot, arm)
    # print(f'\ntool_from_root = {tool_from_root}\n')
    grasp_pose = multiply(pose.value, invert(grasp.value))
    approach_pose = multiply(pose.value, invert(grasp.approach))

    # Gripper Bias
    gripper_bias_quat = quaternion_from_euler(0, 0, math.pi/4, axes='sxyz')
    gripper_bias = ((0, 0, 0), tuple(gripper_bias_quat))

    # grasp_pose = multiply(grasp_pose, gripper_bias)
    # approach_pose = multiply(approach_pose, gripper_bias)

    for tool_pose in interpolate_poses(grasp_pose, approach_pose):
        set_pose(gripper, multiply(tool_pose, tool_from_root, gripper_bias))
        if body is not None:
            set_pose(body, multiply(tool_pose, grasp.value))
        yield

def get_ir_sampler(problem, custom_limits={}, max_attempts=25, collisions=True, learned=True):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    gripper = problem.get_gripper()

    def gen_fn(arm, obj, pose, grasp):
        pose.assign()
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        for _ in iterate_approach_path(robot, arm, gripper, pose, grasp, body=obj):
            if any(pairwise_collision(gripper, b) or pairwise_collision(obj, b) for b in approach_obstacles):
                return
        gripper_pose = multiply(pose.value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        default_conf = arm_conf(arm, grasp.carry)
        arm_joints = get_arm_joints(robot, arm)
        base_joints = get_group_joints(robot, 'base')
        if learned:
            base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp.grasp_type)
        else:
            base_generator = uniform_pose_generator(robot, gripper_pose)
        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        while True:
            count = 0
            for base_conf in islice(base_generator, max_attempts):
                count += 1
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = Conf(robot, base_joints, base_conf)
                pose.assign()
                bq.assign()
                set_joint_positions(robot, arm_joints, default_conf)
                if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                    continue
                yield (bq,)
                break
            else:
                yield None
    return gen_fn

##################################################

# Inverse Kinematics
def get_ik_fn(problem, custom_limits={}, collisions=True, teleport=False):
    obstacles = list(set(problem.fixed).difference(problem.marks)) if collisions else []

    def fn(arm, obj, pose, grasp):

        robot = arm
        arm_link = get_gripper_link(robot, arm)
        arm_joints = get_arm_joints(robot, arm)

        ##################################################
        # Get Obstacles
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        # print(f'\napproach_obstacles = {approach_obstacles}\n')
        default_conf = arm_conf(arm, grasp.carry)

        ##################################################
        # Check Grasp
        # grasp_pose = multiply(pose.value, invert(grasp.value)) 
        # default_conf = arm_conf(arm, grasp.carry)
        # # print(f'\n type of pose = {type(pose)}')
        # # type of pose = <class 'examples.pybullet.utils.pybullet_tools.panda_primitives.Pose'>
        # #sample_fn = get_sample_fn(robot, arm_joints)
        # pose.assign()
        # # base_conf.assign()
        # open_arm(robot, arm)
        # set_joint_positions(robot, arm_joints, default_conf) 
        # grasp_conf = panda_inverse_kinematics(robot, arm, grasp_pose, custom_limits=custom_limits) #, upper_limits=USE_CURRENT)
        # if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles): # [obj]
        #     print('Grasp failed.')
        #     set_joint_positions(robot, arm_joints, default_conf)
        #     return None
        
        ######################################################################
        # Grasp IK
        open_arm(robot, arm)
        pose.assign()
        grasp_pose = multiply(pose.value, invert(grasp.value))
        for i in range(2):
            grasp_conf = panda_inverse_kinematics(robot, arm, grasp_pose, custom_limits=custom_limits) #, upper_limits=USE_CURRENT)
            if grasp_conf is not None:
                break
        if grasp_conf is None:
            print(' No ik solution for grasp pose.')
            return None
        for b in obstacles:
            if pairwise_collision(robot, b):
                print(f' Grasp collision: {robot} and {b}.')
                set_joint_positions(robot, arm_joints, default_conf)
                return None
        # if any(pairwise_collision(robot, b) for b in obstacles): 
        #     print(' Grasp collision.')
        #     set_joint_positions(robot, arm_joints, default_conf)
        #     return None

        ##################################################
        # Check Approach
        approach_pose = multiply(pose.value, invert(grasp.approach))
        approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
            print('Approach failed.')
            set_joint_positions(robot, arm_joints, default_conf)
            return None
        approach_conf = get_joint_positions(robot, arm_joints)
    
        ##################################################
        # Check Grasp Path (Approcach --> Grasp)
        attachment = grasp.get_attachment(robot, arm)
        attachment.child_initial_pose = pose
        attachments = {attachment.child: attachment}
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = 0.05**np.ones(len(arm_joints))
            # grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
            #                                       obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
            #                                       custom_limits=custom_limits, resolutions=resolutions/2.)
            grasp_path = [approach_conf, grasp_conf]
            if grasp_path is None:
                print('Grasp path failure')
                set_joint_positions(robot, arm_joints, default_conf)
                return None
            
        ##################################################
        # Check Approach Path (Default --> Approach)
            set_joint_positions(robot, arm_joints, default_conf)
            approach_path = plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions,
                                              restarts=4, iterations=25, smooth=25) # 25
            if approach_path is None:
                print('Approach path failure')
                set_joint_positions(robot, arm_joints, default_conf)
                return None
            path = approach_path + grasp_path

        ##################################################
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        set_joint_positions(robot, arm_joints, default_conf)
        return (cmd,)
    
    return fn

##################################################

def replanner(arm, obj, pose, grasp, obstacles, custom_limits={}, collisions=True):

    obstacles = list({obst for obst in obstacles if not is_placement(obj, obst)})
    

    robot = arm

    # print(f'\n robot = {robot} \n')
    # print(f'\n arm = {arm} \n')

    arm_link = get_gripper_link(robot, arm)
    
    arm_joints = get_arm_joints(robot, arm)

    default_conf = arm_conf(arm, grasp.carry)
    #sample_fn = get_sample_fn(robot, arm_joints)
    pose.assign()
    # base_conf.assign()
    open_arm(robot, arm)
    set_joint_positions(robot, arm_joints, default_conf) # default_conf | sample_fn()
    
    ######################################################################
    # Grasp IK
    grasp_pose = multiply(pose.value, invert(grasp.value))
    for i in range(2):
        grasp_conf = panda_inverse_kinematics(robot, arm, grasp_pose, custom_limits=custom_limits) #, upper_limits=USE_CURRENT)
        if grasp_conf is not None:
            break
    if grasp_conf is None:
        print(' No ik solution for grasp pose.')
        return None
    if any(pairwise_collision(robot, b) for b in obstacles): 
        input(' Grasp collision.')
        set_joint_positions(robot, arm_joints, default_conf)
        return None

    ######################################################################
    # Approach IK
    approach_pose = multiply(pose.value, invert(grasp.approach))
    approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits) # sub_ik is for near ik from the current pose
    if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
        print(' Approach failed.')
        set_joint_positions(robot, arm_joints, default_conf)
        return None
    approach_conf = get_joint_positions(robot, arm_joints)
    attachment = grasp.get_attachment(robot, arm)
    attachment.child_initial_pose = pose
    attachments = {attachment.child: attachment}
    


    ######################################################################
    # Grasp Path (Approcach --> Grasp)
    resolutions = 0.05**np.ones(len(arm_joints))
    
    grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                            obstacles=[], self_collisions=SELF_COLLISIONS,
                                            custom_limits=custom_limits, resolutions=resolutions/2.)
    if grasp_path is None:
        print(' Grasp path failure')
        set_joint_positions(robot, arm_joints, default_conf)
        return None
    
    ######################################################################
    # Approach Path (Default --> Approach)
    set_joint_positions(robot, arm_joints, default_conf)
    approach_path = plan_joint_motion(robot, arm_joints, list(approach_conf), attachments=attachments.values(),
                                        obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                        custom_limits=custom_limits, resolutions=resolutions,
                                        restarts=2, iterations=25, smooth=25) # 25
    if approach_path is None:
        print(' Approach path failure')
        set_joint_positions(robot, arm_joints, default_conf)
        return None
    path = approach_path + grasp_path

    mt = create_trajectory(robot, arm_joints, path)
    cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
    set_joint_positions(robot, arm_joints, default_conf)
    return (cmd,)

##################################################

def get_ik_ir_gen(problem, max_attempts=2, teleport=False, **kwargs):
    # TODO: compose using general fn
    
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)
    def gen(*inputs):
        b, a, p, g = inputs
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
           
            ik_outputs = ik_fn(*(inputs))
            if ik_outputs is None:
                continue
            print('IK attempts:', attempts)
            # yield ir_outputs + ik_outputs
            
            yield ik_outputs
            return
            #if not p.init:
            #    return
    return gen

##################################################

def get_motion_gen(problem, custom_limits={}, collisions=True, teleport=False):
    # TODO: include fluents
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    def fn(bq1, bq2, fluents=[]):
        saver.restore()
        bq1.assign()
        if teleport:
            path = [bq1, bq2]
        elif is_drake_pr2(robot):
            raw_path = plan_joint_motion(robot, bq2.joints, bq2.values, attachments=[],
                                         obstacles=obstacles, custom_limits=custom_limits, self_collisions=SELF_COLLISIONS,
                                         restarts=4, iterations=50, smooth=50)
            if raw_path is None:
                print('Failed motion plan!')
                #set_renderer(True)
                #for bq in [bq1, bq2]:
                #    bq.assign()
                #    wait_if_gui()
                return None
            path = [Conf(robot, bq2.joints, q) for q in raw_path]
        else:
            goal_conf = base_values_from_pose(bq2.value)
            raw_path = plan_base_motion(robot, goal_conf, BASE_LIMITS, obstacles=obstacles)
            if raw_path is None:
                print('Failed motion plan!')
                return None
            path = [Pose(robot, pose_from_base_values(q, bq1.value)) for q in raw_path]
        bt = Trajectory(path)
        cmd = Commands(State(), savers=[BodySaver(robot)], commands=[bt])
        return (cmd,)
    return fn

##################################################

def get_press_gen(problem, max_attempts=25, learned=True, teleport=False):
    robot = problem.robot
    fixed = get_fixed_bodies(problem)

    def gen(arm, button):
        fixed_wo_button = list(filter(lambda b: b != button, fixed))
        pose = get_pose(button)
        grasp_type = 'side'

        link = get_gripper_link(robot, arm)
        default_conf = get_carry_conf(arm, grasp_type)
        joints = get_arm_joints(robot, arm)

        presses = get_x_presses(button)
        approach = ((APPROACH_DISTANCE, 0, 0), unit_quat())
        while True:
            for _ in range(max_attempts):
                press_pose = random.choice(presses)
                gripper_pose = multiply(pose, invert(press_pose)) # w_f_g = w_f_o * (g_f_o)^-1
                #approach_pose = gripper_pose # w_f_g * g_f_o * o_f_a = w_f_a
                approach_pose = multiply(gripper_pose, invert(multiply(press_pose, approach)))

                if learned:
                    base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp_type)
                else:
                    base_generator = uniform_pose_generator(robot, gripper_pose)
                set_joint_positions(robot, joints, default_conf)
                set_pose(robot, next(base_generator))
                raise NotImplementedError('Need to change this')
                if any(pairwise_collision(robot, b) for b in fixed):
                    continue

                approach_movable_conf = sub_inverse_kinematics(robot, joints[0], link, approach_pose)
                #approach_movable_conf = inverse_kinematics(robot, link, approach_pose)
                if (approach_movable_conf is None) or any(pairwise_collision(robot, b) for b in fixed):
                    continue
                approach_conf = get_joint_positions(robot, joints)

                gripper_movable_conf = sub_inverse_kinematics(robot, joints[0], link, gripper_pose)
                #gripper_movable_conf = inverse_kinematics(robot, link, gripper_pose)
                if (gripper_movable_conf is None) or any(pairwise_collision(robot, b) for b in fixed_wo_button):
                    continue
                grasp_conf = get_joint_positions(robot, joints)
                bp = Pose(robot, get_pose(robot)) # TODO: don't use this

                if teleport:
                    path = [default_conf, approach_conf, grasp_conf]
                else:
                    control_path = plan_direct_joint_motion(robot, joints, approach_conf,
                                                     obstacles=fixed_wo_button, self_collisions=SELF_COLLISIONS)
                    if control_path is None: continue
                    set_joint_positions(robot, joints, approach_conf)
                    retreat_path = plan_joint_motion(robot, joints, default_conf,
                                                     obstacles=fixed, self_collisions=SELF_COLLISIONS)
                    if retreat_path is None: continue
                    path = retreat_path[::-1] + control_path[::-1]
                mt = Trajectory(Conf(robot, joints, q) for q in path)
                yield (bp, mt)
                break
            else:
                yield None
    return gen

#####################################

def control_commands(commands, **kwargs):
    wait_if_gui('Control?')
    disable_real_time()
    enable_gravity()
    for i, command in enumerate(commands):
        print(i, command)
        command.control(*kwargs)

class State(object):
    def __init__(self, attachments={}, cleaned=set(), cooked=set()):
        self.poses = {body: Pose(body, get_pose(body))
                      for body in get_bodies() if body not in attachments}
        self.grasps = {}
        self.attachments = attachments
        self.cleaned = cleaned
        self.cooked = cooked
    def assign(self):
        for attachment in self.attachments.values():
            #attach.attachment.assign()
            attachment.assign()

def apply_commands(state, commands, time_step=None, pause=False, **kwargs):
    for i, command in enumerate(commands):
        print(i, command)
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

#####################################

def get_target_point(conf):
    # TODO: use full body aabb
    robot = conf.body
    link = link_from_name(robot, 'torso_lift_link')
    #link = BASE_LINK
    # TODO: center of mass instead?
    # TODO: look such that cone bottom touches at bottom
    # TODO: the target isn't the center which causes it to drift
    with BodySaver(conf.body):
        conf.assign()
        lower, upper = get_aabb(robot, link)
        center = np.average([lower, upper], axis=0)
        point = np.array(get_group_conf(conf.body, 'base'))
        #point[2] = upper[2]
        point[2] = center[2]
        #center, _ = get_center_extent(conf.body)
        return point


def get_target_path(trajectory):
    # TODO: only do bounding boxes for moving links on the trajectory
    return [get_target_point(conf) for conf in trajectory.path]




################################################
def motion_planner(start_config, start_pose=None, end_pose=None, end_config=None, obstacles=[], teleport=False): # The goal pose shold be in the world space
    # Build pybullet env
    file_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, file_path + '/../../../../../')
    from common.scn import Scenario
    p_pre = get_client()
    p_now = pb.connect(pb.DIRECT)
    set_client(p_now)



    scn = Scenario(bullet_client=p_now)
    obstacles.append(scn.floor)
    robot = scn.add_robot(pose=((0,0,0), (0,0,0,1))) 
    ik_fn = get_simple_ik_fn(robot, fixed=obstacles, teleport=teleport)
    free_motion_fn = get_free_motion_gen(robot, fixed=obstacles, teleport=teleport, algorithm='rrt')

    # Start config
    if start_pose != None:
        conf0 = ik_fn(pose=start_pose)
    else:
        conf0 = BodyConf(robot, configuration=start_config)

    # End config
    if end_pose != None:
        conf1 = ik_fn(pose=end_pose)
    else:
        conf1 = BodyConf(robot, configuration=end_config)


    if conf1 == None:
        print('No IK found.')
        return 

    # Plan
    result = free_motion_fn(conf0, conf1)
    if result == None:
        return None
    path, = result
    pb.disconnect(p_now)
    set_client(p_pre)

    
    return path.body_paths[0].path #CommandMP(path.body_paths)