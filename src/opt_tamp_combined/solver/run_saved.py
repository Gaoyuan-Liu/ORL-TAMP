
import os, sys, time
import pybullet as pb
import pandas as pd
import pickle

from opt_solver import Solver
from execute import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../')
sys.path.insert(0, file_path + '/../../pddlstream/')

from pddlstream.language.constants import Equal, And

from common.scn import Scenario

from examples.pybullet.utils.pybullet_tools.utils import connect
import pybullet as p

RUN_TIMES = 1
def main():

    connect(use_gui=True)

    #################################################
    # Scenario 
    scn = Scenario()
    scn.reset()
    robot = scn.add_robot(pose=((0,0,0), (0,0,0,1)))
    table_low = scn.add_low_table()
    table_high = scn.add_high_table()
    plate = scn.add_plate(pose=((0.45, 0.15, 0.31),(0,0,0,1)))
    bar = scn.add_bar(((0.6, -0.1, 0.33), (0,0,0,1)))
    cup = scn.add_cup(((0.8, 0.1, 0.33), (0,0,0,1)))
    pb.changeDynamics(cup, -1, lateralFriction=0.8)


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
    goal = And(('on', scn.plate, table_high), ('on', cup, scn.plate))
    
    #################################################
    # Solve the problem
 
    # plan, cost, evaluations = solver.solve(goal)
    solver.problem_formation()
    plan = pickle.load(open("./plan.pkl", "rb", -1))

   
    
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