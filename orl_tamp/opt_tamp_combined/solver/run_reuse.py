
import os, sys, time
import pybullet as pb
import pandas as pd
import pickle
file_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, file_path)
from opt_solver import Solver
from execute import Executor



sys.path.insert(0, file_path + '/../../pddlstream/')
from pddlstream.language.constants import Equal, And

sys.path.insert(0, file_path + '/../../')
from utils.scn import add_ground, add_robot, add_cylinder, add_bar, add_cup, add_table
from utils.pybullet_tools.utils import connect

RUN_TIMES = 1

def main():

    mode = sys.argv[1]


    #################################################

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
    pb.changeDynamics(plate, -1, lateralFriction=0.1)

    bar = add_bar(((0.6, -0.1, 0.33), (0,0,0,1)))
    scn['bar'] = bar

    cup = add_cup(((0.8, 0.05, 0.33), (0,0,0,1)))
    scn['cup'] = cup
    



    # Print
    print(' \n====== Scenario ====== ')
    print(' Robot: ', robot)
    print(' Low Table: ', table_low)
    print(' High Table: ', table_high)
    print(' plate: ', plate)
    print(' Bar: ', bar)
    print(' Cup: ', cup)
    print(' ====================== \n')
    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)


    input('\n Scene is ready. Press Enter to plan. ')

    #################################################
    # Solve the problem
    solver = Solver(robot, movable=[plate, cup], tools=[bar], surfaces=[table_low, table_high, plate],
                    top_grasps=[cup], edge_grasps=[plate])
    executor = Executor(scn)

    #################################################
    # Define the goal
    

    # goal = And(('on', scn.plate, table_high), ('on', cup, scn.plate))
    
    #################################################
    # Solve the problem
 
    # plan, cost, evaluations = solver.solve(goal)
    solver.problem_formation()
    if mode == "edgepush":
        plan = pickle.load(open("./plans/plan_edgepush.pkl", "rb", -1))
    elif mode == "retrieve":
        plan = pickle.load(open("./plans/plan_retrieve.pkl", "rb", -1))
    elif mode == "rearrange":
        plan = pickle.load(open("./plans/plan_rearrange.pkl", "rb", -1))
    else:
        print("Invalid mode")

   
    
    if plan is None:
        print('Planning failed')
        return
    else:

        input('\n Plan is ready. Press Enter to execute. ')
        executor.execute(solver.problem, plan)

def reset(scn):
    pass




if __name__ == '__main__':
    main()