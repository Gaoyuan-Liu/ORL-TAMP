from doctest import master
from multiprocessing.reduction import ForkingPickler
from platform import node
from random import random

from .utils import irange, argmin, RRT_ITERATIONS
import roboticstoolbox as rtb
import numpy as np
from math import dist


class TreeNode(object):

    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent

    #def retrace(self):
    #    if self.parent is None:
    #        return [self]
    #    return self.parent.retrace() + [self]

    def retrace(self): # This can retrace to the beginning
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def clear(self):
        self.node_handle = None
        self.edge_handle = None

    def draw(self, env, color=(1, 0, 0, .5)):
        from manipulation.primitives.display import draw_node, draw_edge
        self.node_handle = draw_node(env, self.config, color=color)
        if self.parent is not None:
            self.edge_handle = draw_edge(
                env, self.config, self.parent.config, color=color)

    def __str__(self):
        return 'TreeNode(' + str(self.config) + ')'
    __repr__ = __str__


def configs(nodes):
    if nodes is None:
        return None
    return list(map(lambda n: n.config, nodes))


def rrt(start, goal_sample, distance, sample, extend, collision, goal_test=lambda q: False,
        iterations=RRT_ITERATIONS, goal_probability=.2):
    if collision(start):
        return None
    if not callable(goal_sample):
        g = goal_sample
        goal_sample = lambda: g
    nodes = [TreeNode(start)]
    for i in irange(iterations):
        goal = random() < goal_probability or i == 0 # There is a small change that does not sample but choose goal as the next sample
        s = goal_sample() if goal else sample()

        # print(f'\n s = {s} \b')

        last = argmin(lambda n: distance(n.config, s), nodes) # If collid it will not start from the beginning
        i = 0
        for q in extend(last.config, s):
            if collision(q):
                break
            last = TreeNode(q, parent=last)
            nodes.append(last)
            if goal_test(last.config):
                return configs(last.retrace())
        else:
            if goal:
                return configs(last.retrace())
    return None


def rrt_sub(start, goal_sample, distance, sample, extend, collision, collision_sub, master_path, goal_test=lambda q: False,
        iterations=1000, goal_probability=.1): #RRT_ITERATIONS = 20
    if collision(start):
        return None
    if not callable(goal_sample):
        g = goal_sample
        goal_sample = lambda: g
    nodes = [TreeNode(start)]

    
    for i in irange(iterations):
        # print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
        goal = random() < goal_probability or i == 0 # There is a small change that does not sample but choose goal as the next sample
        s = goal_sample() if goal else sample()

        # print(f'\n s = {s} \b')

        last = argmin(lambda n: distance(n.config, s), nodes) # If collid it will not start from the beginning
        # last = nodes[-1]
        # if len(last.retrace()) > len(master_path):
        #     master_path_long = master_path + [master_path[-1]]*(len(last.retrace())-len(master_path))
        # else:
        #     master_path_long = master_path
        i = 0
        for q in extend(last.config, s):
            # print(f'\n len = {len(last.retrace())} \n')
            if collision(q):
                break
            # print(f'\n master_path = {last.retrace()} \n')
            
            if len(master_path) <= len(last.retrace()):
                master_current = master_path[-1]
            else:
                master_current = master_path[len(last.retrace())-1]
            if collision_sub(master_current, q):
                last = last.retrace()[-10]
                del nodes[-10:]
                break
            last = TreeNode(q, parent=last)
            nodes.append(last)
            if goal_test(last.config):
                return configs(last.retrace())
        else:
            if goal:
                return configs(last.retrace())
    return None


def rrt_dual(start1, goal_sample1, ee_goal1, start2, goal_sample2, ee_goal2, distance, sample, extend, collision, collision_sub,
        iterations=1000, goal_probability=.9): #RRT_ITERATIONS = 20
    if collision(start1):
        print('Start Pose of Robot1 is in collision.')
        return None
    if collision(start2):
        print('Start Pose of Robot2 is in collision.')
        return None

    if not callable(goal_sample1):
        print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
        g1 = goal_sample1
        goal_sample1 = lambda: g1

    if not callable(goal_sample2):
        print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
        g2 = goal_sample2
        goal_sample2 = lambda: g2

    nodes1 = [TreeNode(start1)]
    nodes2 = [TreeNode(start2)]

    robot_1_done = False
    robot_2_done = False
    DONE = False

    # last1 = nodes1[0]
    # last2 = nodes2[0]
    for i in irange(iterations):

        print('New Iteration!!!!!!!!!!!!!!!!!!')
        goal1 = random() < goal_probability or i == 0 # There is a small change that does not sample but choose goal as the next sample
        s1 = goal_sample1() if goal1 else sample()

        goal2 = random() < goal_probability or i == 0 # There is a small change that does not sample but choose goal as the next sample
        s2 = goal_sample2() if goal2 else sample()


        last1 = argmin(lambda n: distance(n.config, s1), nodes1) # If collid it will not start from the beginning
        last2 = argmin(lambda m: distance(m.config, s2), nodes2)


        q1_index_after_last = 0
        q2_index_after_last = 0

        # if robot_1_done == False and robot_2_done == False:
        for q1 in extend(last1.config, s1):
            # print('here again')
            q1_index_after_last += 1
            if collision(q1):
                break
            last1 = TreeNode(q1, parent=last1)
            nodes1.append(last1)
            robot_1_done = goal_test(ee_goal1, last1.config, (0, 0, 0))
            if robot_1_done:
                robot_1_path = configs(last1.retrace())
                print('Robot 1 is done!')
                break
        
        
        for q2 in extend(last2.config, s2):
            q2_index_after_last += 1
            if collision(q2):
                break
            if len(last2.retrace()) <= len(robot_1_path):
                q1_current = robot_1_path[len(last2.retrace())-1] # -1 for retrace length and -1 for the last start from 1
            else: 
                q1_current = robot_1_path[-1]
            # print('\n')
            # print(len(last2.retrace()) - 1)
            # print(len(robot_1_path))
            if collision_sub(q1_current, q2):
                break
            last2 = TreeNode(q2, parent=last2)
            nodes2.append(last2)
            robot_2_done = goal_test(ee_goal2, last2.config, (0,1,0)) # Robot2 has bias!!!!!!!!!!!!!!!!
            if robot_2_done:
                print('robot2 is done!')
                robot_2_path = configs(last2.retrace())
                # if len(robot_2_path) < len(robot_1_path)
                break
    
        if robot_1_done and robot_2_done:
            # print('All done!')
            break

    if robot_2_done and (len(robot_1_path) > len(robot_2_path)):

        print('Re-adjust robot 1.')
        robot_1_done = False
        
        save_node = last1.retrace()[len(robot_2_path)] 
        nodes1 = [last1.retrace()[len(robot_2_path)]]
        # print(f'length of a new nodes is {nodes1}')
        for i in irange(iterations):
            goal1 = random() < goal_probability or i == 0 
            s1 = goal_sample1() if goal1 else sample()
            last1 = argmin(lambda n: distance(n.config, s1), nodes1)

        
            print('Re-adjust robot 1.')
    
            del robot_1_path[len(robot_2_path):]
         
            for q1 in extend(last1.config, s1):
                if collision(q1):
                    break
                if collision_sub(q1, robot_2_path[-1]):
                    break
                last1 = TreeNode(q1, parent=last1)
                nodes1.append(last1)
                robot_1_done = goal_test(ee_goal1, last1.config, (0, 0, 0))
                if robot_1_done:
                    robot_1_path = configs(last1.retrace())
                    print('Robot 1 is done again!')
                    break

            if robot_1_done:
                print('All done!')
                break
                    





    return {'robot1':robot_1_path, 'robot2':robot_2_path}#{'robot1':configs(last1.retrace()), 'robot2':configs(last2.retrace())}
        # else:
        #     if goal1 and goal2:
        #         return {'robot1':configs(last1.retrace()), 'robot2':configs(last2.retrace())}
    # return None

def goal_test(cartesian_goal, q, robot_base):
    robot = rtb.models.Panda()
    ee_position = np.array(robot.fkine_all(q)[9])[:3,3]+ np.array(robot_base)
    if dist(cartesian_goal[0], ee_position) < 0.01:
        return True
    # Te = type(Te)
    else:
        return False



# Robot 2 wait at the waypoint where no robot 1 waypoint can collid 
# They should do a mutual collision checking
def rrt_wait(start1, goal_sample1, ee_goal1, start2, goal_sample2, ee_goal2, distance, sample, extend, collision, collision_sub,
        iterations=1000, goal_probability=.9): #RRT_ITERATIONS = 20
    if collision(start1):
        print('Start Pose of Robot1 is in collision.')
        return None
    if collision(start2):
        print('Start Pose of Robot2 is in collision.')
        return None

    if not callable(goal_sample1):
        g1 = goal_sample1
        goal_sample1 = lambda: g1

    if not callable(goal_sample2):
        g2 = goal_sample2
        goal_sample2 = lambda: g2

    nodes1 = [TreeNode(start1)]
    nodes2 = [TreeNode(start2)]

    robot_1_done = False
    robot_2_done = False
    DONE = False

    # last1 = nodes1[0]
    # last2 = nodes2[0]
    for i in irange(iterations):

        print('New Iteration!!!!!!!!!!!!!!!!!!')
        goal1 = random() < goal_probability or i == 0 # There is a small change that does not sample but choose goal as the next sample
        s1 = goal_sample1() if goal1 else sample()

        goal2 = random() < goal_probability or i == 0 # There is a small change that does not sample but choose goal as the next sample
        s2 = goal_sample2() if goal2 else sample()


        last1 = argmin(lambda n: distance(n.config, s1), nodes1) # If collid it will not start from the beginning
        last2 = argmin(lambda m: distance(m.config, s2), nodes2)


        q1_index_after_last = 0
        q2_index_after_last = 0

        # Plan robot 1
        for q1 in extend(last1.config, s1):
            # print('here again')
            q1_index_after_last += 1
            if collision(q1):
                break
            last1 = TreeNode(q1, parent=last1)
            nodes1.append(last1)
            robot_1_done = goal_test(ee_goal1, last1.config, (0, 0, 0))
            if robot_1_done:
                robot_1_path = configs(last1.retrace())
                print('Robot 1 is done!')
                break
        
        # Plan robot 2
        for q2 in extend(last2.config, s2):
            q2_index_after_last += 1
            if collision(q2):
                break
            last2 = TreeNode(q2, parent=last2)
            nodes2.append(last2)
            robot_2_done = goal_test(ee_goal2, last2.config, (0,1,0)) # Robot2 has bias!!!!!!!!!!!!!!!!
            if robot_2_done:
                print('robot2 is done!')
                robot_2_path = configs(last2.retrace())
                break
    
    # Find the waiting waypoint for robot 1
    # The waiting way point should be the last point which will not be collid with any of the waypoint
    # The priority should be the one with shorter trajectory
    if len(robot_1_path) <= len(robot_2_path):
        priority = 1
    else:
        priority = 2


    for i in range(len(robot_1_path)):
        collid = False
        # Go through all the robot2 waypoints
        for robot_2_waypoint in robot_2_path:
            if collision_sub(robot_1_path[i], robot_2_waypoint) == True:
                collid = True
                break
        if collid == True:
            waiting_1 = i
        else:
            break

    for j in range(len(robot_2_path)):
        collid = False
        # Go through all the robot2 waypoints
        for robot_1_waypoint in robot_1_path:
            if collision_sub(robot_1_waypoint, robot_2_path[j]) == True:
                collid = True
                break
        if collid == True:
            waiting_2 = j
        else:
            break

    return {'robot1':robot_1_path, 'robot2':robot_2_path}, priority, waiting_1, waiting_2 
          




       

    
                


    # Find the waiting waypoint for robot 2
   
