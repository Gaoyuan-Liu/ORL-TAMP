
import os, sys, time
import pybullet as pb
import pandas as pd
import pickle

from opt_solver import Solver
from execute import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../')

from pddlstream.language.constants import Equal, And

sys.path.insert(0, file_path + '/../../')
from utils.scn import add_ground, add_robot, add_cylinder, add_bar, add_cup, add_table
from utils.pybullet_tools.utils import connect


RUN_TIMES = 1
def main():

    connect(use_gui=True)

    #################################################
    # Scenario 
    scn = {}

    ground = add_ground()
    scn['ground'] = ground

    robot = add_robot(pose=((0,0,0), (0,0,0,1)))
    scn['robot'] = robot

    table_low = add_table(1, 0.5, 0.15, pose=((0.75, 0, 0.075),(0,0,0,1))) #add_low_table()
    scn['low_table'] = table_low

    table_high = add_table(0.5, 0.5, 0.4, pose=((0.5, -0.5, 0.2),(0,0,0,1))) 
    scn['high_table'] = table_high

    plate = add_cylinder(0.08, 0.03, pose=((0.45, 0.15, 0.31),(0,0,0,1)), color=(0.6,0.6,0.6,1))
    scn['plate'] = plate

    bar = add_bar(((0.6, -0.1, 0.33), (0,0,0,1)))
    scn['bar'] = bar

    cup = add_cup(((0.8, 0.05, 0.33), (0,0,0,1)))
    scn['cup'] = cup


    # Print
    print(' \n====== Scenario ====== ')
    print(' scn = ', scn)
    print(' ====================== \n')
    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)


    input('\n Scene is ready. Press Enter to plan. ')

    #################################################
    # Solve the problem
    solver = Solver(robot, movable=[scn['plate'], scn['cup']], surfaces=[scn['low_table'], scn['high_table'], scn['plate']],
                    top_grasps=[scn['cup']], edge_grasps=[scn['plate']])
    executor = Executor(scn)

    #################################################
    # Define the goal
    # goal = And(('on', scn['plate'], table_high), ('on', scn['cup'], scn['plate']))
    # goal = ('on', scn['cup'], scn['high_table'])
    goal = ('on', scn['plate'], scn['high_table'])
    #################################################
    # Solve the problem
 
    plan, cost, evaluations = solver.solve(goal)

   
    
    if plan is None:
        print('Planning failed')
        return
    else:
        # with open('./plans/plan_rearrange.pkl', 'wb') as file:
        #     pickle.dump(plan, file)

        input('\n Plan is ready. Press Enter to execute. ')
        executor.execute(solver.problem, plan)

def reset(scn):
    pass




if __name__ == '__main__':
    main()